#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
import argparse
import yaml
from PIL import ExifTags, Image, ImageOps

from multiprocessing.pool import Pool


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)


# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def get_hash(paths):
    """Get the hash value of paths"""
    assert isinstance(paths, list), "Only support list currently."
    h = hashlib.md5("".join(paths).encode())
    return h.hexdigest()


def check_image(im_file):
    '''Verify an image.'''
    nc, msg = 0, ""
    try:
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = im.size  # (width, height)
        try:
            im_exif = im._getexif()
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])
        except:
            im_exif = None
        if im_exif and ORIENTATION in im_exif:
            rotation = im_exif[ORIENTATION]
            if rotation in (6, 8):
                shape = (shape[1], shape[0])

        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
        return im_file, shape, nc, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
        return im_file, None, nc, msg


def check_label_files(args):
    img_path, lb_path = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
    try:
        if osp.exists(lb_path):
            nf = 1  # label found
            with open(lb_path, "r") as f:
                labels = [
                    x.split() for x in f.read().strip().splitlines() if len(x)
                ]
                labels = np.array(labels, dtype=np.float32)
            if len(labels):
                assert all(
                    len(l) == 5 for l in labels
                ), f"{lb_path}: wrong label format."
                assert (
                    labels >= 0
                ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                assert (
                    labels[:, 1:] <= 1
                ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                _, indices = np.unique(labels, axis=0, return_index=True)
                if len(indices) < len(labels):  # duplicate row check
                    labels = labels[indices]  # remove duplicates
                    msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                labels = labels.tolist()
            else:
                ne = 1  # label empty
                labels = []
        else:
            nm = 1  # label missing
            labels = []

        return img_path, labels, nc, nm, nf, ne, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
        return img_path, None, nc, nm, nf, ne, msg


def generate_coco_format_labels(img_info, class_names, save_path):
    # for evaluation with pycocotools
    dataset = {"categories": [], "annotations": [], "images": []}
    for i, class_name in enumerate(class_names):
        dataset["categories"].append(
            {"id": i, "name": class_name, "supercategory": ""}
        )

    ann_id = 0
    LOGGER.info(f"Convert to COCO format")
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info["labels"] if info["labels"] else []
        img_id = osp.splitext(osp.basename(img_path))[0]
        img_w, img_h = info["shape"]
        dataset["images"].append(
            {
                "file_name": os.path.basename(img_path),
                "id": img_id,
                "width": img_w,
                "height": img_h,
            }
        )
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                # convert x,y,w,h to x1,y1,x2,y2
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h
                # cls_id starts from 0
                cls_id = int(c)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset["annotations"].append(
                    {
                        "area": h * w,
                        "bbox": [x1, y1, w, h],
                        "category_id": cls_id,
                        "id": ann_id,
                        "image_id": img_id,
                        "iscrowd": 0,
                        # mask
                        "segmentation": [],
                    }
                )
                ann_id += 1

    with open(save_path, "w") as f:
        json.dump(dataset, f)
        LOGGER.info(
            f"Convert to COCO format finished. Resutls saved in {save_path}"
        )


def generate_coco_annotations(img_dir, class_names, check_images=False, check_labels=False):
    assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
    valid_img_record = osp.join(
        osp.dirname(img_dir), "." + osp.basename(img_dir) + ".json"
    )
    NUM_THREADS = min(8, os.cpu_count())

    img_paths = glob.glob(osp.join(img_dir, "**/*"), recursive=True)
    img_paths = sorted(
        p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)
    )
    assert img_paths, f"No images found in {img_dir}."

    img_hash = get_hash(img_paths)
    if osp.exists(valid_img_record):
        with open(valid_img_record, "r") as f:
            cache_info = json.load(f)  # {'information':{...}, 'image_hash':...,'label_hash':...}
            if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                img_info = cache_info["information"]
            else:
                check_images = True
    else:
        check_images = True

    # check images
    if check_images:
        img_info = {}
        nc, msgs = 0, []  # number corrupt, messages
        LOGGER.info(
            f"Checking formats of images with {NUM_THREADS} process(es): "
        )
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(check_image, img_paths),
                total=len(img_paths),
            )
            for img_path, shape_per_img, nc_per_img, msg in pbar:
                if nc_per_img == 0:  # not corrupted
                    img_info[img_path] = {"shape": shape_per_img}
                nc += nc_per_img
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{nc} image(s) corrupted"
        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))

        cache_info = {"information": img_info, "image_hash": img_hash}
        # save valid image paths.
        with open(valid_img_record, "w") as f:
            json.dump(cache_info, f)

    # check and load anns
    label_dir = osp.join(
        osp.dirname(osp.dirname(img_dir)), "labels", osp.basename(img_dir)
    )
    assert osp.exists(label_dir), f"{label_dir} is an invalid directory path!"

    # Look for labels in the save relative dir that the images are in
    def _new_rel_path_with_ext(base_path: str, full_path: str, new_ext: str):
        rel_path = osp.relpath(full_path, base_path)
        return osp.join(osp.dirname(rel_path), osp.splitext(osp.basename(rel_path))[0] + new_ext)

    img_paths = list(img_info.keys())
    label_paths = sorted(
        osp.join(label_dir, _new_rel_path_with_ext(img_dir, p, ".txt"))
        for p in img_paths
    )
    assert label_paths, f"No labels found in {label_dir}."
    label_hash = get_hash(label_paths)
    if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
        check_labels = True

    if check_labels:
        cache_info["label_hash"] = label_hash
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
        LOGGER.info(
            f"Checking formats of labels with {NUM_THREADS} process(es): "
        )
        with Pool(NUM_THREADS) as pool:
            pbar = pool.imap(check_label_files, zip(img_paths, label_paths))
            pbar = tqdm(pbar, total=len(label_paths))
            for (
                    img_path,
                    labels_per_file,
                    nc_per_file,
                    nm_per_file,
                    nf_per_file,
                    ne_per_file,
                    msg,
            ) in pbar:
                if nc_per_file == 0:
                    img_info[img_path]["labels"] = labels_per_file
                else:
                    img_info.pop(img_path)
                nc += nc_per_file
                nm += nm_per_file
                nf += nf_per_file
                ne += ne_per_file
                if msg:
                    msgs.append(msg)

                pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"

        pbar.close()
        with open(valid_img_record, "w") as f:
             json.dump(cache_info, f)
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(
                f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
            )


    save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    save_path = osp.join(
        save_dir, "instances_" + osp.basename(img_dir) + ".json"
    )
    generate_coco_format_labels(img_info, class_names, save_path)

    img_paths, labels = list(
        zip(
            *[
                (
                    img_path,
                    np.array(info["labels"], dtype=np.float32)
                    if info["labels"]
                    else np.zeros((0, 5), dtype=np.float32),
                )
                for img_path, info in img_info.items()
            ]
        )
    )

    LOGGER.info(
        f" Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
    )

    return img_paths, labels

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generate COCO Annotations', add_help=add_help)
    parser.add_argument('--img_dir', default='./coco128/images/val2017', type=str, help='path of dataset')
    parser.add_argument('--data_config', default='./data/coco128.yaml', type=str, help='path of dataset')

    return parser


def main(args):
    img_dir = args.img_dir
    data_dict = load_yaml(args.data_config)
    class_names = data_dict['names']
    _, _ = generate_coco_annotations(img_dir, class_names)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)


