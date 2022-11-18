import os
import json
import argparse
import shutil
import instances2dict_with_polygons as cs


def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]

    return box_from_poly


def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)

    return xywh_box


def convert_to_yolo_box(im_size, box):
    # box: [xmin, ymin, xmax, ymax]
    dw = 1. / im_size[0]
    dh = 1. / im_size[1]
    TO_REMOVE = 1
    w = box[2] - box[0] + TO_REMOVE
    h = box[3] - box[1] + TO_REMOVE
    xc = box[0] + w / 2.0
    yc = box[1] + h / 2.0
    xc = xc * dw
    w = w * dw
    yc = yc * dh
    h = h * dh
    return xc, yc, w, h


def make_yolo_dirs(dst_dir):
    dir_domains = ['source', 'target']
    dir_mode = ['images', 'labels']
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for domain in dir_domains:
        dir = os.path.join(dst_dir, domain)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for mode in dir_mode:
            sub_dir = os.path.join(dir, mode)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            sub_dir_train = os.path.join(sub_dir, 'train')
            if not os.path.exists(sub_dir_train):
                os.makedirs(sub_dir_train)
            sub_dir_val = os.path.join(sub_dir, 'val')
            if not os.path.exists(sub_dir_val):
                os.makedirs(sub_dir_val)
            sub_dir_test = os.path.join(sub_dir, 'test')
            if not os.path.exists(sub_dir_test):
                os.makedirs(sub_dir_test)


# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)


category_instancesonly = [
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    ]

category_dict = {
    'person': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
}

def convert_cityscapes_to_yolo(data_dir, yolo_dir, data_type):
    # data_dir: cityscapes root dir
    # yolo_dir: suurce or target domain dir,
    # e.g., ./source |_images|_train
    #                |       |_val
    #                |_labels|_train
    #                        |_val
    # data_type : 'source' or 'target'
    #
    sets = [
        'gtFine_val',
        'gtFine_train',
        'gtFine_test',
    ]
    ann_dirs = [
        'gtFine/val',
        'gtFine/train',
        'gtFine/test',
    ]
    ends_in = '%s_polygons.json'

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)

        ann_dir = os.path.join(data_dir, ann_dir)
        images = []

        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in % data_set.split('_')[0]):

                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    img_size = [image['width'], image['height']]
                    if data_type == 'source':
                        image['file_name'] = filename.split('_')[0] + '/' + \
                                             filename[:-len(ends_in % data_set.split('_')[0])] + 'leftImg8bit.png'
                    elif data_type == 'target':
                        image['file_name'] = filename.split('_')[0] + '/' + \
                                             filename[:-len(
                                                 ends_in % data_set.split('_')[0])] + 'leftImg8bit_foggy_beta_0.02.png'
                    image['seg_file_name'] = filename[:-len(
                                                 ends_in % data_set.split('_')[0])] + '%s_instanceIds.png' % data_set.split('_')[0]
                    fullname = os.path.join(root, image['seg_file_name'])
                    fullname = os.path.abspath(fullname)
                    objects = cs.instances2dict_with_polygons([fullname], verbose=False)[fullname]

                    yolo_objs = []
                    for object_cls in objects:
                        if object_cls not in category_instancesonly:
                            continue # skip non-instance categories
                        for obj in objects[object_cls]:
                            if obj['contours'] == []:
                                print('Warning: empty contours.')
                                continue  # skip non-instance categories
                            len_p = [len(p) for p in obj['contours']]
                            if min(len_p) <= 4:
                                print('Warning: invalid contours.')
                                continue  # skip non-instance categories
                            xyxy_box = poly_to_box(obj['contours'])
                            yolo_box = list(convert_to_yolo_box(img_size, xyxy_box))
                            cat_id = category_dict[object_cls]
                            yolo_box.insert(0, cat_id)
                            yolo_objs.append(yolo_box)
                    image['objects'] = yolo_objs
                    images.append(image)

        ## construct yolo images and labels
        cur_data_set = data_set.split('_')[1]
        file_list = cur_data_set + '.txt'
        file_list = os.path.join(yolo_dir, data_type, file_list)
        with open(file_list, 'w', encoding='utf-8') as f1:
            for cur_image in images:
                cur_image_name = cur_image['file_name']
                if data_type == 'source':
                    image_fullfile = os.path.join(data_dir, 'leftImg8bit', cur_data_set, cur_image_name)
                elif data_type == 'target':
                    image_fullfile = os.path.join(data_dir, 'leftImg8bit_foggy', cur_data_set, cur_image_name)
                dst_image_file = os.path.join('images', cur_data_set, cur_image_name.split('/')[1])
                f1.write(dst_image_file + '\n')
                dst_image_file = os.path.join(yolo_dir, data_type, dst_image_file)
                if not os.path.exists(dst_image_file):
                    shutil.copyfile(os.path.abspath(image_fullfile), os.path.abspath(dst_image_file))
                cur_label_txt = os.path.join(yolo_dir, data_type, 'labels', cur_data_set, cur_image_name.split('/')[1][:-4]+'.txt')
                if not os.path.exists(cur_label_txt):
                    with open(cur_label_txt, 'w', encoding='utf-8') as f2:
                        for cur_obj in cur_image['objects']:
                            obj_to_write = "".join([str(a) + " " for a in cur_obj])
                            obj_to_write.rstrip(" ")
                            f2.write(obj_to_write + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--data_dir', help="data dir for annotations to be converted",default=None, type=str)
    parser.add_argument('--dst_dir', help="data dir for annotations to be converted", default=None, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    dst_dir = args.dst_dir
    #1. construct the uda yolo directors
    make_yolo_dirs(dst_dir)
    #2. construct the source yolo dataset
    convert_cityscapes_to_yolo(data_dir, dst_dir, 'source')
    #2. construct the target yolo dataset
    convert_cityscapes_to_yolo(data_dir, dst_dir, 'target')



