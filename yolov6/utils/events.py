#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = shutil.get_terminal_size().columns


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to log."""
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)

    tblogger.add_scalar("x/lr0", results[2], epoch + 1)
    tblogger.add_scalar("x/lr1", results[3], epoch + 1)
    tblogger.add_scalar("x/lr2", results[4], epoch + 1)


def write_tblog_pr(tblogger, epoch, results):
    """Display pr metric information to log."""
    tblogger.add_scalar("val/precision", results[0], epoch + 1)
    tblogger.add_scalar("val/recall", results[1], epoch + 1)
    tblogger.add_scalar("val/f1-score", results[2], epoch + 1)


def write_tblog_gdc(tblogger, epoch, results, losses, eval_domain='both', mix_wo_gdc=False):
    """Display mAP and loss information to log."""
    if eval_domain !='both':
        tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
        tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)
        tblogger.add_scalar("x/lr0", results[2], epoch + 1)
        tblogger.add_scalar("x/lr1", results[3], epoch + 1)
        tblogger.add_scalar("x/lr2", results[4], epoch + 1)
        if not mix_wo_gdc:
            tblogger.add_scalar("gdc_x/lr0", results[5], epoch + 1)
            tblogger.add_scalar("gdc_x/lr1", results[6], epoch + 1)
            tblogger.add_scalar("gdc_x/lr2", results[7], epoch + 1)
    else:
        tblogger.add_scalar("val_s/mAP@0.5", results[0], epoch + 1)
        tblogger.add_scalar("val_s/mAP@0.50:0.95", results[1], epoch + 1)
        tblogger.add_scalar("val_t/mAP@0.5", results[2], epoch + 1)
        tblogger.add_scalar("val_t/mAP@0.50:0.95", results[3], epoch + 1)
        tblogger.add_scalar("x/lr0", results[4], epoch + 1)
        tblogger.add_scalar("x/lr1", results[5], epoch + 1)
        tblogger.add_scalar("x/lr2", results[6], epoch + 1)
        if not mix_wo_gdc:
            tblogger.add_scalar("gdc_x/lr0", results[7], epoch + 1)
            tblogger.add_scalar("gdc_x/lr1", results[8], epoch + 1)
            tblogger.add_scalar("gdc_x/lr2", results[9], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)
    if mix_wo_gdc:
        tblogger.add_scalar("train/det_t_loss", losses[3], epoch + 1)
    else:
        tblogger.add_scalar("train/dc_s_loss", losses[3], epoch + 1)
        tblogger.add_scalar("train/dc_t_loss", losses[4], epoch + 1)


def write_tblog_gdc_pr(tblogger, epoch, results, eval_domain='both'):
    """Display pr metric information to log."""
    if eval_domain !='both':
        tblogger.add_scalar("val/precision", results[0], epoch + 1)
        tblogger.add_scalar("val/recall", results[1], epoch + 1)
        tblogger.add_scalar("val/f1-score", results[2], epoch + 1)
    else:
        tblogger.add_scalar("val_s/precision", results[0], epoch + 1)
        tblogger.add_scalar("val_s/recall", results[1], epoch + 1)
        tblogger.add_scalar("val_s/f1-score", results[2], epoch + 1)
        tblogger.add_scalar("val_t/precision", results[3], epoch + 1)
        tblogger.add_scalar("val_t/recall", results[4], epoch + 1)
        tblogger.add_scalar("val_t/f1-score", results[5], epoch + 1)


def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')


def write_tbimg_gdc(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        tblogger.add_image(f'train_batch_target', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_target_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')