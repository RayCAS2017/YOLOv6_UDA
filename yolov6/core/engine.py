#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import time
from copy import deepcopy
import os.path as osp

from tqdm import tqdm

import cv2
import numpy as np
import math
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import tools.eval as eval
import tools.eval_gdc as eval_gdc
from yolov6.data.data_load import create_dataloader
from yolov6.models.yolo import build_model
from yolov6.models.global_domain_classifier import build_gdc_net
from yolov6.models.loss import ComputeLoss
from yolov6.models.loss_distill import ComputeLoss as ComputeLoss_distill
from yolov6.models.loss_gdc import ComputeLoss_gdc
from yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog, write_tbimg
from yolov6.utils.events import write_tblog_gdc, write_tbimg_gdc, write_tblog_pr, write_tblog_gdc_pr
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from yolov6.utils.checkpoint import save_checkpoint_best_map_0_5
from yolov6.utils.checkpoint import strip_optimizer_gdc
from yolov6.solver.build import build_optimizer, build_lr_scheduler
from yolov6.solver.build import build_lr_scheduler_gdc, build_optimizer_gdc
from yolov6.utils.RepOptimizer import extract_scales, RepVGGOptimizer
from yolov6.utils.nms import xywh2xyxy


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir
        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']
        if args.gdc:
            self.train_loader_s, self.val_loader_s, self.train_loader_t, self.val_loader_t = self.get_data_loader_gdc(args, cfg, self.data_dict)
        else:
            self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict)
        # get model and optimizer
        model = self.get_model(args, cfg, self.num_classes, device)
        if args.gdc and not args.mix_train_wo_gdc:
            gdc_net = self.get_gdc_net(args, cfg, device)
        if self.args.distill:
            self.teacher_model = self.get_teacher_model(args, cfg, self.num_classes, device)
        if self.args.quant:
            self.quant_setup(model, cfg, device)
        if cfg.training_mode == 'repopt':
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
        else:
            self.optimizer = self.get_optimizer(args, cfg, model)
            if args.gdc and not args.mix_train_wo_gdc:
                self.optimizer_gdc = self.get_optimizer_gdc(args, cfg, gdc_net)
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        if args.gdc and not args.mix_train_wo_gdc:
            self.scheduler_gdc, self.lf_gdc = self.get_lr_scheduler_gdc(args, cfg, self.optimizer_gdc)
        self.ema = ModelEMA(model) if self.main_process else None
        if args.gdc and not args.mix_train_wo_gdc:
            self.ema_gdc = ModelEMA(gdc_net) if self.main_process else None
        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
        self.start_epoch = 0
        #resume
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            if args.gdc and not args.mix_train_wo_gdc:
                resume_state_dict_gdc = self.ckpt['gdc_net'].float().state_dict()
                gdc_net.load_state_dict(resume_state_dict_gdc, strict=True)
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if args.gdc and not args.mix_train_wo_gdc:
                self.optimizer_gdc.load_state_dict(self.ckpt['optimizer_gdc'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']
                if args.gdc and not args.mix_train_wo_gdc:
                    self.ema_gdc.ema.load_state_dict(self.ckpt['ema_gdc'].float().state_dict())
                    self.ema_gdc.updates = self.ckpt['updates_gdc']

        if args.gdc and not args.mix_train_wo_gdc:
            self.model = self.parallel_model_gdc(args, model, device)
            self.gdc_net = self.parallel_model_gdc(args, gdc_net, device)
        else:
            self.model = self.parallel_model(args, model, device)

        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

        self.max_epoch = args.epochs
        if args.gdc:
            self.max_stepnum = len(self.train_loader_s)
        else:
            self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.vis_imgs_list = []
        self.vis_imgs_list_2 = []
        self.write_trainbatch_tb = args.write_trainbatch_tb
        # set color for classnames
        self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]

        self.loss_num = 3
        self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss']
        if self.args.distill:
            self.loss_num += 1
            self.loss_info += ['cwd_loss']
        if self.args.gdc and not self.args.mix_train_with_gdc and not self.args.mix_train_wo_gdc:
            self.loss_num += 2
            self.loss_info += ['ds_loss', 'dt_loss']
        elif self.args.gdc and self.args.mix_train_with_gdc:
            self.loss_num += 3
            self.loss_info += ['ds_loss', 'dt_loss', 'det_t_loss']
        elif self.args.gdc and self.args.mix_train_wo_gdc:
            self.loss_num += 1
            self.loss_info += ['det_t_loss']

        if args.gdc and args.gdc_eval_domain:
            self.gdc_eval_domain = args.gdc_eval_domain

        # self.save_best_map_0_5 = args.save_best_ap_0_5

        # self.do_pr_metric = args.do_pr_metric

    # Training Process
    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop(self.epoch)
            self.strip_model()
            
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    # Training loop for each epoch
    def train_in_loop(self, epoch_num):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                if self.args.gdc:
                    try:
                        self.batch_data_t = next(self.pbar_t)
                    except StopIteration:
                        self.pbar_t = iter(self.train_loader_t)
                        self.batch_data_t = next(self.pbar_t)
                self.train_in_steps(epoch_num)
                self.print_details()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise
        try:
            self.eval_and_save()
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    # Training loop for batchdata
    def train_in_steps(self, epoch_num):
        images, targets = self.prepro_data(self.batch_data, self.device)
        if self.args.gdc:
            if self.args.mix_train_with_gdc or self.args.mix_train_wo_gdc:
                images_t, targets_t = self.prepro_data(self.batch_data_t, self.device)
            else:
                images_t, _ = self.prepro_data(self.batch_data_t, self.device)
        # plot train_batch and save to tensorboard once an epoch
        if self.write_trainbatch_tb and self.main_process and self.step == 0:
            self.plot_train_batch(images, targets)
            write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')
            if self.args.gdc:
                self.plot_train_batch_t(images_t)
                write_tbimg_gdc(self.tblogger, self.vis_train_batch_t, self.step + self.max_stepnum * self.epoch, type='train')

        # forward
        with amp.autocast(enabled=self.device != 'cpu'):
            preds, s_featmaps, b_featmaps = self.model(images)
            # s_featmaps, student featmat for distill
            # b_feat, backbone featmat, p3, p4, p5
            if self.args.distill:
                with torch.no_grad():
                    t_preds, t_featmaps = self.teacher_model(images)
                temperature = self.args.temperature
                total_loss, loss_items = self.compute_loss_distill(preds, t_preds, s_featmaps, t_featmaps, targets, \
                                                                   epoch_num, self.max_epoch, temperature)
            elif self.args.gdc and not self.args.mix_train_with_gdc and not self.args.mix_train_wo_gdc:
                # 1: source domain detector training
                loss_det_s, loss_items = self.compute_loss(preds, targets, epoch_num)

                # 2: source domain global domain classifier training
                gdc_logits_s = self.gdc_net(b_featmaps)
                domain_target = torch.tensor(1, device=self.device, dtype=gdc_logits_s[0].dtype)
                loss_gdc_s = self.compute_loss_gdc(gdc_logits_s, domain_target)
                loss_items = torch.cat((loss_items, loss_gdc_s.clone().detach().unsqueeze(0)), 0)

                # 3: target domain global domain classifier trainning
                _, _, b_featmaps_t = self.model(images_t)
                gdc_logits_t = self.gdc_net(b_featmaps_t)
                domain_target = torch.tensor(0, device=self.device, dtype=gdc_logits_t[0].dtype)
                loss_gdc_t = self.compute_loss_gdc(gdc_logits_t, domain_target)
                loss_items = torch.cat((loss_items, loss_gdc_t.clone().detach().unsqueeze(0)), 0)  # ['iou', 'dfl', 'class', 'gdc_s', 'gdc_t']

                total_loss = loss_det_s + loss_gdc_s + loss_gdc_t

            elif self.args.gdc and self.args.mix_train_with_gdc:
                # 1: source domain detector training
                loss_det_s, loss_items = self.compute_loss(preds, targets, epoch_num)

                # 2: source domain global domain classifier training
                gdc_logits_s = self.gdc_net(b_featmaps)
                domain_target = torch.tensor(1, device=self.device, dtype=gdc_logits_s[0].dtype)
                loss_gdc_s = self.compute_loss_gdc(gdc_logits_s, domain_target)
                loss_items = torch.cat((loss_items, loss_gdc_s.clone().detach().unsqueeze(0)), 0)

                # 3: target domain detector training
                preds_t, _, b_featmaps_t = self.model(images_t)
                loss_det_t, _ = self.compute_loss(preds_t, targets_t, epoch_num)

                # 4: target domain global domain classifier trainning
                gdc_logits_t = self.gdc_net(b_featmaps_t)
                domain_target = torch.tensor(0, device=self.device, dtype=gdc_logits_t[0].dtype)
                loss_gdc_t = self.compute_loss_gdc(gdc_logits_t, domain_target)
                loss_items = torch.cat((loss_items, loss_gdc_t.clone().detach().unsqueeze(0)), 0)  # ['iou', 'dfl', 'class', 'gdc_s', 'gdc_t']
                loss_items = torch.cat((loss_items, loss_det_t.clone().detach().unsqueeze(0)), 0)  # ['iou', 'dfl', 'class', 'gdc_s', 'gdc_t', 'det_t_loss']
                total_loss = loss_det_s + loss_det_t + loss_gdc_s + loss_gdc_t

            elif self.args.gdc and self.args.mix_train_wo_gdc:
                # 1: source domain detector training
                loss_det_s, loss_items = self.compute_loss(preds, targets, epoch_num)

                # 2: target domain detector training
                preds_t, _, b_featmaps_t = self.model(images_t)
                loss_det_t, _ = self.compute_loss(preds_t, targets_t, epoch_num)
                loss_items = torch.cat((loss_items, loss_det_t.clone().detach().unsqueeze(0)), 0)

                total_loss = loss_det_s + loss_det_t

            else:
                total_loss, loss_items = self.compute_loss(preds, targets, epoch_num)
            if self.rank != -1:
                total_loss *= self.world_size
        # backward

        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def eval_and_save(self):
        remaining_epochs = self.max_epoch - self.epoch
        eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 1
        is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
        if self.main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride']) # update attributes for ema model
            if is_val_epoch:
                self.eval_model()
                if self.args.gdc:
                    self.ap = self.evaluate_results[3]
                    self.best_ap = max(self.ap, self.best_ap)
                    self.ap_0_5 = self.evaluate_results[2]
                    self.best_ap_0_5 = max(self.ap_0_5, self.best_ap_0_5)
                else:
                    self.ap = self.evaluate_results[1]
                    self.best_ap = max(self.ap, self.best_ap)
                    self.ap_0_5 = self.evaluate_results[0]
                    self.best_ap_0_5 = max(self.ap_0_5, self.best_ap_0_5)

            # save ckpt
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'gdc_net': deepcopy(de_parallel(self.gdc_net)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'ema_gdc': deepcopy(self.ema_gdc.ema).half(),
                    'updates': self.ema.updates,
                    'updates_gdc': self.ema_gdc.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'optimizer_gdc': self.optimizer_gdc.state_dict(),
                    'epoch': self.epoch,
                }
            else:
                ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                }

            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            if self.args.save_best_ap_0_5 and self.ap_0_5 == self.best_ap_0_5 and is_val_epoch:
                save_checkpoint_best_map_0_5(save_ckpt_dir)
            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')

            #default save best ap ckpt in stop strong aug epochs
            if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
                if self.best_stop_strong_aug_ap < self.ap:
                    self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
                    save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')
                
            del ckpt
            # log for learning rate
            lr = [x['lr'] for x in self.optimizer.param_groups] 
            self.evaluate_results = list(self.evaluate_results) + lr
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                lr_gdc = [x['lr'] for x in self.optimizer_gdc.param_groups]
                self.evaluate_results += lr_gdc
            
            # log for tensorboard
            if self.args.gdc:
                write_tblog_gdc(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss, eval_domain=self.gdc_eval_domain, mix_wo_gdc=self.args.mix_train_wo_gdc)
                if self.args.do_pr_metric:
                    write_tblog_gdc_pr(self.tblogger, self.epoch, list(self.evaluate_pr_metric_results), eval_domain=self.gdc_eval_domain)
            else:
                write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)
                write_tblog_pr(self.tblogger, self.epoch, list(self.evaluate_pr_metric_results))
            # save validation predictions to tensorboard
            write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')
            if self.args.gdc and self.gdc_eval_domain == 'both':
                write_tbimg_gdc(self.tblogger, self.vis_imgs_list_2, self.epoch, type='val')
            
    def eval_model(self):
        if not hasattr(self.cfg, "eval_params"):
            if self.args.gdc:
                if self.gdc_eval_domain == 'source':
                    dataloader_gdc = [self.val_loader_s, None]
                elif self.gdc_eval_domain == 'target':
                    dataloader_gdc = [self.val_loader_t, None]
                elif self.gdc_eval_domain == 'both':
                    dataloader_gdc = [self.val_loader_s, self.val_loader_t]
                else:
                    LOGGER.warning('gdc_eval_domain should be source, target, or both.\n')

                if self.args.do_pr_metric:
                    results, vis_outputs, vis_paths, pr_results = eval_gdc.run(self.data_dict,
                                                                   batch_size=self.batch_size // self.world_size * 2,
                                                                   img_size=self.img_size,
                                                                   model=self.ema.ema if self.args.calib is False else self.model,
                                                                   conf_thres=0.03,
                                                                   dataloader=dataloader_gdc,
                                                                   save_dir=self.save_dir,
                                                                   task='train',
                                                                   domain=self.gdc_eval_domain,
                                                                   do_pr_metric=self.args.do_pr_metric)
                else:
                    results, vis_outputs, vis_paths = eval_gdc.run(self.data_dict,
                                                                   batch_size=self.batch_size // self.world_size * 2,
                                                                   img_size=self.img_size,
                                                                   model=self.ema.ema if self.args.calib is False else self.model,
                                                                   conf_thres=0.03,
                                                                   dataloader=dataloader_gdc,
                                                                   save_dir=self.save_dir,
                                                                   task='train',
                                                                   domain=self.gdc_eval_domain)


                if self.gdc_eval_domain != 'both':
                    LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
                    self.evaluate_results[:2] = results[:2]
                    if self.args.do_pr_metric:
                        self.evaluate_pr_metric_results[:3] = pr_results[:3]
                    # plot validation predictions
                    self.plot_val_pred(vis_outputs, vis_paths)
                else:
                    LOGGER.info(f"Epoch: {self.epoch} | mAP_s@0.5: {results[0][0]} | mAP_s@0.50:0.95: {results[0][1]} | mAP_t@0.5: {results[1][0]} | mAP_t@0.50:0.95: {results[1][1]}")
                    self.evaluate_results[:2] = results[0][:2]
                    self.evaluate_results[2:] = results[1][:2]
                    if self.args.do_pr_metric:
                        self.evaluate_pr_metric_results[:3] = pr_results[0][:3]
                        self.evaluate_pr_metric_results[3:] = pr_results[1][:3]
                    # source
                    self.plot_val_pred(vis_outputs[0], vis_paths[0])
                    # target
                    self.plot_val_pred_2(vis_outputs[1], vis_paths[1])
            else:
                if self.args.do_pr_metric:
                    results, vis_outputs, vis_paths, pr_results = eval.run(self.data_dict,
                                                               batch_size=self.batch_size // self.world_size * 2,
                                                               img_size=self.img_size,
                                                               model=self.ema.ema if self.args.calib is False else self.model,
                                                               conf_thres=0.03,
                                                               dataloader=self.val_loader,
                                                               save_dir=self.save_dir,
                                                               task='train',
                                                               do_pr_metric=self.args.do_pr_metric)
                else:
                    results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                               batch_size=self.batch_size // self.world_size * 2,
                                                               img_size=self.img_size,
                                                               model=self.ema.ema if self.args.calib is False else self.model,
                                                               conf_thres=0.03,
                                                               dataloader=self.val_loader,
                                                               save_dir=self.save_dir,
                                                               task='train')


                LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
                self.evaluate_results = results[:2]
                if self.args.do_pr_metric:
                    self.evaluate_pr_metric_results[:3] = pr_results[:3]
                # plot validation predictions
                self.plot_val_pred(vis_outputs, vis_paths)

        else:
            def get_cfg_value(cfg_dict, value_str, default_value):
                if value_str in cfg_dict and cfg_dict[value_str] is not None:
                    return cfg_dict[value_str]
                else:
                    return default_value
            eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
            if self.args.gdc:
                if self.gdc_eval_domain == 'source':
                    dataloader_gdc = [self.val_loader_s, None]
                elif self.gdc_eval_domain == 'target':
                    dataloader_gdc = [self.val_loader_t, None]
                elif self.gdc_eval_domain == 'both':
                    dataloader_gdc = [self.val_loader_s, self.val_loader_s]
                else:
                    LOGGER.warning('gdc_eval_domain should be source, target, or both.\n')

                if self.args.do_pr_metric:
                    results, vis_outputs, vis_paths, pr_results = eval_gdc.run(self.data_dict,
                                                                   batch_size=get_cfg_value(self.cfg.eval_params,self.batch_size // self.world_size * 2),
                                                                   img_size=eval_img_size, model=self.ema.ema if self.args.calib is False else self.model,
                                                                   conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                                                                   dataloader=dataloader_gdc,
                                                                   save_dir=self.save_dir,
                                                                   task='train',
                                                                   test_load_size=get_cfg_value(self.cfg.eval_params, "test_load_size", eval_img_size),
                                                                   letterbox_return_int=get_cfg_value(self.cfg.eval_params, "letterbox_return_int", False),
                                                                   force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                                                                   not_infer_on_rect=get_cfg_value(self.cfg.eval_params,  "not_infer_on_rect", False),
                                                                   scale_exact=get_cfg_value(self.cfg.eval_params,"scale_exact", False),
                                                                   verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                                   do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                                                                   do_pr_metric=self.args.do_pr_metric,
                                                                   plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                                                                   plot_confusion_matrix=get_cfg_value( self.cfg.eval_params, "plot_confusion_matrix",  False),
                                                                   domain=self.gdc_eval_domain)
                else:
                    results, vis_outputs, vis_paths = eval_gdc.run(self.data_dict,
                                                                   batch_size=get_cfg_value(self.cfg.eval_params, self.batch_size // self.world_size * 2),
                                                                   img_size=eval_img_size,
                                                                   model=self.ema.ema if self.args.calib is False else self.model,
                                                                   conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                                                                   dataloader=dataloader_gdc,
                                                                   save_dir=self.save_dir,
                                                                   task='train',
                                                                   test_load_size=get_cfg_value(self.cfg.eval_params, "test_load_size", eval_img_size),
                                                                   letterbox_return_int=get_cfg_value(self.cfg.eval_params, "letterbox_return_int",False),
                                                                   force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                                                                   not_infer_on_rect=get_cfg_value(self.cfg.eval_params, "not_infer_on_rect", False),
                                                                   scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact", False),
                                                                   verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                                   do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                                                                   do_pr_metric= False,
                                                                   plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                                                                   plot_confusion_matrix=get_cfg_value(self.cfg.eval_params, "plot_confusion_matrix",False),
                                                                   domain=self.gdc_eval_domain)

                if self.gdc_eval_domain != 'both':
                    LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
                    self.evaluate_results[:2] = results[:2]
                    if self.args.do_pr_metric:
                        self.evaluate_pr_metric_results[:3] = pr_results[:3]
                    # plot validation predictions
                    self.plot_val_pred(vis_outputs, vis_paths)
                else:
                    LOGGER.info(f"Epoch: {self.epoch} | mAP_s@0.5: {results[0][0]} | mAP_s@0.50:0.95: {results[0][1]} | mAP_t@0.5: {results[1][0]} | mAP_t@0.50:0.95: {results[1][1]}")
                    self.evaluate_results[:2] = results[0][:2]
                    self.evaluate_results[2:] = results[1][:2]

                    if self.args.do_pr_metric:
                        self.evaluate_pr_metric_results[:3] = pr_results[0][:3]
                        self.evaluate_pr_metric_results[3:] = pr_results[1][:3]
                    # source
                    self.plot_val_pred(vis_outputs[0], vis_paths[0])
                    # target
                    self.plot_val_pred_2(vis_outputs[1], vis_paths[1])

            else:
                if self.args.do_pr_metric:
                    results, vis_outputs, vis_paths, pr_results = eval.run(self.data_dict,
                                                               batch_size=get_cfg_value(self.cfg.eval_params, "batch_size",self.batch_size // self.world_size * 2),
                                                               img_size=eval_img_size,
                                                               model=self.ema.ema if self.args.calib is False else self.model,
                                                               conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                                                               dataloader=self.val_loader,
                                                               save_dir=self.save_dir,
                                                               task='train',
                                                               test_load_size=get_cfg_value(self.cfg.eval_params, "test_load_size", eval_img_size),
                                                               letterbox_return_int=get_cfg_value(self.cfg.eval_params, "letterbox_return_int",  False),
                                                               force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                                                               not_infer_on_rect=get_cfg_value(self.cfg.eval_params,  "not_infer_on_rect",  False),
                                                               scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact", False),
                                                               verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                               do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                                                               do_pr_metric=self.args.do_pr_metric,
                                                               plot_curve=get_cfg_value(self.cfg.eval_params,  "plot_curve", False),
                                                               plot_confusion_matrix=get_cfg_value(self.cfg.eval_params, "plot_confusion_matrix", False),
                                                               )
                else:
                    results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                           batch_size=get_cfg_value(self.cfg.eval_params, "batch_size",self.batch_size // self.world_size * 2),
                                                           img_size=eval_img_size,
                                                           model=self.ema.ema if self.args.calib is False else self.model,
                                                           conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                                                           dataloader=self.val_loader,
                                                           save_dir=self.save_dir,
                                                           task='train',
                                                           test_load_size=get_cfg_value(self.cfg.eval_params, "test_load_size", eval_img_size),
                                                           letterbox_return_int=get_cfg_value(self.cfg.eval_params, "letterbox_return_int", False),
                                                           force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                                                           not_infer_on_rect=get_cfg_value(self.cfg.eval_params, "not_infer_on_rect", False),
                                                           scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact", False),
                                                           verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                           do_coco_metric=get_cfg_value(self.cfg.eval_params, "do_coco_metric", True),
                                                           do_pr_metric=False,
                                                           plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                                                           plot_confusion_matrix=get_cfg_value(self.cfg.eval_params, "plot_confusion_matrix", False),
                                                           )
                LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
                self.evaluate_results = results[:2]
                if self.args.do_pr_metric:
                    self.evaluate_pr_metric_results[:3] = pr_results[:3]
                # plot validation predictions
                self.plot_val_pred(vis_outputs, vis_paths)

    def train_before_loop(self):
        LOGGER.info('Training start...')
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum), 1000) if self.args.quant is False else 0
        self.scheduler.last_epoch = self.start_epoch - 1
        if self.args.gdc and not self.args.mix_train_wo_gdc:
            self.scheduler_gdc.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.last_opt_setp_gdc = -1
        self.last_opt_setp_gdc_s = -1
        self.last_opt_setp_gdc_t = -1

        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

        if self.args.gdc:
            self.best_ap, self.ap = 0.0, 0.0
            self.best_ap_0_5, self.ap_0_5 = 0.0, 0.0
            self.best_stop_strong_aug_ap = 0.0
            self.evaluate_results = [0, 0, 0, 0]  # AP50_s, AP50_95_s, AP50_t, AP50_95_t
            self.evaluate_pr_metric_results = [0, 0, 0, 0, 0, 0] # p_s, r_s, f1_s, p_t, r_t, f1_t
        else:
            self.best_ap, self.ap = 0.0, 0.0
            self.best_ap_0_5, self.ap_0_5 = 0.0, 0.0
            self.best_stop_strong_aug_ap = 0.0
            self.evaluate_results = (0, 0)  # AP50, AP50_95
            self.evaluate_pr_metric_results = [0, 0, 0]  # p, r, f1

        self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        use_dfl=self.cfg.model.head.use_dfl,
                                        reg_max=self.cfg.model.head.reg_max,
                                        iou_type=self.cfg.model.head.iou_type)
        if self.args.gdc and not self.args.mix_train_wo_gdc:
            self.compute_loss_gdc = ComputeLoss_gdc(loss_weight=self.cfg.gdc.loss_weight)

        if self.args.distill:                             
            self.compute_loss_distill = ComputeLoss_distill(num_classes=self.data_dict['nc'],
                                                            ori_img_size=self.img_size,
                                                            use_dfl=self.cfg.model.head.use_dfl,
                                                            reg_max=self.cfg.model.head.reg_max,
                                                            iou_type=self.cfg.model.head.iou_type,
                                                            distill_weight = self.cfg.model.head.distill_weight,
                                                            distill_feat = self.args.distill_feat,
                                                            )

    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                self.scheduler_gdc.step()
        #stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            if self.args.gdc:
                self.train_loader_s, self.val_loader_s, self.train_loader_t, self.val_loader_t = self.get_data_loader_gdc(self.args, self.cfg, self.data_dict)
            else:
                self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)

        self.model.train()
        if self.args.gdc and not self.args.mix_train_wo_gdc:
            self.gdc_net.train()
        if self.rank != -1:
            if self.args.gdc:
                self.train_loader_s.sampler.set_epoch(self.epoch)
                self.train_loader_t.sampler.set_epoch(self.epoch)
            else:
                self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(self.loss_num, device=self.device)
        self.optimizer.zero_grad()
        if self.args.gdc and not self.args.mix_train_wo_gdc:
            self.optimizer_gdc.zero_grad()

        LOGGER.info(('\n' + '%10s' * (self.loss_num + 1)) % (*self.loss_info,))
        if self.args.gdc:
            self.pbar = enumerate(self.train_loader_s)
            self.max_iter_one_epoch = len(self.train_loader_s)
            self.pbar_t = iter(self.train_loader_t)
        else:
            self.pbar = enumerate(self.train_loader)

        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            self.pbar.set_description(('%10s' + '%10.4g' * self.loss_num) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                *(self.mean_loss)))

    def strip_model(self):
        if self.main_process:
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                strip_optimizer_gdc(save_ckpt_dir, self.epoch)
            else:
                strip_optimizer(save_ckpt_dir, self.epoch)  # strip optimizers for saved pt model

    # Empty cache if training finished
    def train_after_loop(self):
        if self.device != 'cpu':
            torch.cuda.empty_cache()

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum], [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum], [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                for k, param in enumerate(self.optimizer_gdc.param_groups):
                    warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                    param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                            [warmup_bias_lr, param['initial_lr'] * self.lf_gdc(self.epoch)])
                    if 'momentum' in param:
                        param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                      [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])

        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                self.scaler.step(self.optimizer_gdc)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.args.gdc and not self.args.mix_train_wo_gdc:
                self.optimizer_gdc.zero_grad()
            if self.ema:
                self.ema.update(self.model)
                if self.args.gdc and not self.args.mix_train_wo_gdc:
                    self.ema_gdc.update(self.gdc_net)
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        train_path, val_path = data_dict['train'], data_dict['val']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
        # create val dataloader
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='val')[0]

        return train_loader, val_loader

    @staticmethod
    def get_data_loader_gdc(args, cfg, data_dict):
        train_path_s, val_path_s, train_path_t, val_path_t = data_dict['train_s'], data_dict['val_s'], data_dict['train_t'], data_dict['val_t']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader_s = create_dataloader(train_path_s, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
        train_loader_t = create_dataloader(train_path_t, args.img_size, args.batch_size // args.world_size, grid_size,
                                           hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                           workers=args.workers, shuffle=True, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
        # create val dataloader
        val_loader_s, val_loader_t = None, None
        if args.rank in [-1, 0]:
            val_loader_s = create_dataloader(val_path_s, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='val')[0]
            val_loader_t = \
            create_dataloader(val_path_t, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                              hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                              workers=args.workers, check_images=args.check_images,
                              check_labels=args.check_labels, data_dict=data_dict, task='val')[0]

        return train_loader_s, val_loader_s, train_loader_t, val_loader_t

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        targets = batch_data[1].to(device)
        return images, targets

    def get_model(self, args, cfg, nc, device):
        model = build_model(cfg, nc, device)
        weights = cfg.model.pretrained
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)

        LOGGER.info('Model: {}'.format(model))
        return model

    def get_gdc_net(self, args, cfg, device):
        assert 'gdc' in cfg.keys(), 'the gdc dict is not in the config file'
        gdc_net = build_gdc_net(cfg, device)
        return gdc_net
        
    def get_teacher_model(self, args,cfg,nc, device):
        model = build_model(cfg, nc, device)
        weights = args.teacher_model_path
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f'Loading state_dict from {weights} for teacher')
            model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        # Do not update running means and running vars
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
        return model

    @staticmethod
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.model.scales
        scales = None
        if not weights:
            LOGGER.error("ERROR: No scales provided to init RepOptimizer!")
        else:
            ckpt = torch.load(weights, map_location=device)
            scales = extract_scales(ckpt)
        return scales

    @staticmethod
    def parallel_model(args, model, device):
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)

        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        return model

    @staticmethod
    def parallel_model_gdc(args, model, device):
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)
        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            '''
            # https://www.wangt.cc/2021/06/one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/
            # https://discuss.pytorch.org/t/ddp-sync-batch-norm-gradient-computation-modified/82847/5
            # for fixing bug: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation [2021-12-14]
            '''
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

        return model

    def get_optimizer(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        optimizer = build_optimizer(cfg, model)
        return optimizer

    def get_optimizer_gdc(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        optimizer = build_optimizer_gdc(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf

    @staticmethod
    def get_lr_scheduler_gdc(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler_gdc(cfg, optimizer, epochs)
        return lr_scheduler, lf

    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        paths = self.batch_data[2]  # image paths
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = xywh2xyxy(ti[:, 2:6]).T
                classes = ti[:, 1].astype('int')
                labels = ti.shape[1] == 6  # labels if no conf column
                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = self.data_dict['names'][cls] if self.data_dict['names'] else cls
                    if labels:
                        label = f'{cls}'
                        cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                        cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, thickness=1)
        self.vis_train_batch = mosaic.copy()

    def plot_train_batch_t(self, images, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        paths = self.batch_data_t[2]  # image paths
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename
        self.vis_train_batch_t = mosaic.copy()

    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        # plot validation predictions
        self.vis_imgs_list = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()     # xyxy
            ori_img = cv2.imread(vis_path)
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]), thickness=1)
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))

    def plot_val_pred_2(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        # plot validation predictions
        self.vis_imgs_list_2 = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()     # xyxy
            ori_img = cv2.imread(vis_path)
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]), thickness=1)
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            self.vis_imgs_list_2.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))

    # PTQ
    def calibrate(self, cfg):
        def save_calib_model(model, cfg):
            # Save calibrated checkpoint
            output_model_path = os.path.join(cfg.ptq.calib_output_path, '{}_calib_{}.pt'.
                                             format(os.path.splitext(os.path.basename(cfg.model.pretrained))[0], cfg.ptq.calib_method))
            if cfg.ptq.sensitive_layers_skip is True:
                output_model_path = output_model_path.replace('.pt', '_partial.pt')
            LOGGER.info('Saving calibrated model to {}... '.format(output_model_path))
            if not os.path.exists(cfg.ptq.calib_output_path):
                os.mkdir(cfg.ptq.calib_output_path)
            torch.save({'model': deepcopy(de_parallel(model)).half()}, output_model_path)
        assert self.args.quant is True and self.args.calib is True
        if self.main_process:
            from tools.qat.qat_utils import ptq_calibrate
            ptq_calibrate(self.model, self.train_loader, cfg)
            self.epoch = 0
            self.eval_model()
            save_calib_model(self.model, cfg)
    # QAT
    def quant_setup(self, model, cfg, device):
        if self.args.quant:
            from tools.qat.qat_utils import qat_init_model_manu, skip_sensitive_layers
            qat_init_model_manu(model, cfg, self.args)
            # workaround
            model.neck.upsample_enable_quant()
            # if self.main_process:
            #     print(model)
            # QAT
            if self.args.calib is False:
                if cfg.qat.sensitive_layers_skip:
                    skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)
                # QAT flow load calibrated model
                assert cfg.qat.calib_pt is not None, 'Please provide calibrated model'
                model.load_state_dict(torch.load(cfg.qat.calib_pt)['model'].float().state_dict())
            model.to(device)
