#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import os.path as osp
import sys
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.evaler_gdc import Evaler
from yolov6.utils.events import LOGGER
from yolov6.utils.general import increment_name
from yolov6.utils.config import Config

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Evalating', add_help=add_help)
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='./weights/yolov6s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.03, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='val, or speed')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help='whether to use fp16 infer')
    parser.add_argument('--save_dir', type=str, default='runs/val/', help='evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help='save evaluation results to save_dir/name')
    parser.add_argument('--test_load_size', type=int, default=640, help='load img resize when test')
    parser.add_argument('--letterbox_return_int', default=False, action='store_true', help='return int offset for letterbox')
    parser.add_argument('--scale_exact', default=False, action='store_true', help='use exact scale size to scale coords')
    parser.add_argument('--force_no_pad', default=False, action='store_true', help='for no extra pad in letterbox')
    parser.add_argument('--not_infer_on_rect', default=False, action='store_true', help='default to use rect image size to boost infer')
    parser.add_argument('--reproduce_640_eval', default=False, action='store_true', help='whether to reproduce 640 infer result, overwrite some config')
    parser.add_argument('--eval_config_file', type=str, default='./configs/experiment/eval_640_repro.py', help='config file for repro 640 infer result')
    parser.add_argument('--do_coco_metric', default=True, action='store_true', help='whether to use pycocotool to metric, set False to close')
    parser.add_argument('--do_pr_metric', default=False, action='store_true', help='whether to calculate precision, recall and F1, n, set False to close')
    parser.add_argument('--plot_curve', default=True, action='store_true', help='whether to save plots in savedir when do pr metric, set False to close')
    parser.add_argument('--plot_confusion_matrix', default=False, action='store_true', help='whether to save confusion matrix plots when do pr metric, might cause no harm warning print')
    parser.add_argument('--verbose', default=False, action='store_true', help='whether to print metric on each class')
    parser.add_argument('--domain', default='source', help='source, or target, or both')
    args = parser.parse_args()

    # load params for reproduce 640 eval result
    if args.reproduce_640_eval:
        assert os.path.exists(args.eval_config_file), print("Reproduce config file {} does not exist".format(args.eval_config_file))
        eval_params = Config.fromfile(args.eval_config_file).eval_params
        eval_model_name = os.path.splitext(os.path.basename(args.weights))[0]
        if eval_model_name not in eval_params:
            eval_model_name = "default"
        args.test_load_size = eval_params[eval_model_name]["test_load_size"]
        args.letterbox_return_int = eval_params[eval_model_name]["letterbox_return_int"]
        args.scale_exact = eval_params[eval_model_name]["scale_exact"]
        args.force_no_pad = eval_params[eval_model_name]["force_no_pad"]
        args.not_infer_on_rect = eval_params[eval_model_name]["not_infer_on_rect"]

    LOGGER.info(args)
    return args


@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        img_size=640,
        conf_thres=0.03,
        iou_thres=0.65,
        task='val',
        device='',
        half=False,
        model=None,
        dataloader=[None, None],
        save_dir='',
        name = '',
        test_load_size=640,
        letterbox_return_int=False,
        force_no_pad=False,
        not_infer_on_rect=False,
        scale_exact=False,
        reproduce_640_eval=False,
        eval_config_file='./configs/experiment/eval_640_repro.py',
        verbose=False,
        do_coco_metric=True,
        do_pr_metric=False,
        plot_curve=False,
        plot_confusion_matrix=False,
        domain='source'
        ):
    """ Run the evaluation process

    This function is the main process of evaluataion, supporting image file and dir containing images.
    It has tasks of 'val', 'train' and 'speed'. Task 'train' processes the evaluation during training phase.
    Task 'val' processes the evaluation purely and return the mAP of model.pt. Task 'speed' precesses the
    evaluation of inference speed of model.pt.

    """

     # task
    Evaler.check_task(task)
    if task == 'train':
        save_dir = save_dir
    else:
        save_dir = str(increment_name(osp.join(save_dir, name)))
        os.makedirs(save_dir, exist_ok=True)

    # check the threshold value, reload device/half/data according task
    Evaler.check_thres(conf_thres, iou_thres, task)
    device = Evaler.reload_device(device, model, task)
    half = device.type != 'cpu' and half
    data = Evaler.reload_dataset(data) if isinstance(data, str) else data

    # init
    if domain != 'both':
        val = Evaler(data, batch_size, img_size, conf_thres, \
                       iou_thres, device, half, save_dir, \
                       test_load_size, letterbox_return_int, force_no_pad, not_infer_on_rect, scale_exact,
                       verbose, do_coco_metric, do_pr_metric, plot_curve, plot_confusion_matrix, domain=domain)
        model, _ = val.init_model(model, weights, task)
        dataloader[0] = val.init_data(dataloader[0], task) # if task=='train', the input dataloader = [dataload_domain, None]
    else:
        val_s = Evaler(data, batch_size, img_size, conf_thres, \
                        iou_thres, device, half, save_dir, \
                        test_load_size, letterbox_return_int, force_no_pad, not_infer_on_rect, scale_exact,
                        verbose, do_coco_metric, do_pr_metric, plot_curve, plot_confusion_matrix, domain='source')
        model, stride = val_s.init_model(model, weights, task)
        val_t = Evaler(data, batch_size, img_size, conf_thres, \
                        iou_thres, device, half, save_dir, \
                        test_load_size, letterbox_return_int, force_no_pad, not_infer_on_rect, scale_exact,
                        verbose, do_coco_metric, do_pr_metric, plot_curve, plot_confusion_matrix, domain='target')
        val_t.set_stide(stride)
        dataloader[0] = val_s.init_data(dataloader[0], task)  # if task=='train', the input dataloader = [dataload_s, dataload_t]
        dataloader[1] = val_t.init_data(dataloader[1], task)

    # eval
    model.eval()
    if domain != 'both':
        if do_pr_metric:
            pred_result, vis_outputs, vis_paths, pr_results = val.predict_model(model, dataloader[0], task) # pr_resuots: [precision, recall, F1]
        else:
            pred_result, vis_outputs, vis_paths = val.predict_model(model, dataloader[0], task)
        eval_result = val.eval_model(pred_result, model, dataloader[0], task)

        if do_pr_metric:
            return eval_result, vis_outputs, vis_paths, pr_results
        else:
            return eval_result, vis_outputs, vis_paths
    else:
        eval_result_both, vis_outputs_both, vis_paths_both = [], [], []
        if do_pr_metric:
            pr_results_both = []
            pred_result_s, vis_outputs_s, vis_paths_s, pr_results_s = val_s.predict_model(model, dataloader[0], task)
            pr_results_both.append(pr_results_s)
        else:
            pred_result_s, vis_outputs_s, vis_paths_s = val_s.predict_model(model, dataloader[0], task)
        eval_result_s = val_s.eval_model(pred_result_s, model, dataloader[0], task)
        eval_result_both.append(eval_result_s)
        vis_outputs_both.append(vis_outputs_s)
        vis_paths_both.append(vis_paths_s)
        if do_pr_metric:
            pred_result_t, vis_outputs_t, vis_paths_t, pr_results_t = val_t.predict_model(model, dataloader[1], task)
            pr_results_both.append(pr_results_t)
        else:
            pred_result_t, vis_outputs_t, vis_paths_t = val_t.predict_model(model, dataloader[1], task)
        eval_result_t = val_t.eval_model(pred_result_t, model, dataloader[1], task)
        eval_result_both.append(eval_result_t)
        vis_outputs_both.append(vis_outputs_t)
        vis_paths_both.append(vis_paths_t)

        if do_pr_metric:
            return eval_result_both, vis_outputs_both, vis_paths_both, pr_results_both
        else:
            return eval_result_both, vis_outputs_both, vis_paths_both


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
