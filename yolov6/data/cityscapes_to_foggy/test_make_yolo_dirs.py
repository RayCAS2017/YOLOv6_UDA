import os

def make_yolo_dirs(dst_dir):
    dir_domains = ['source', 'target']
    dir_mode = ['images', 'labels']
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

if __name__ == '__main__':
    dst_dir = './'
    make_yolo_dirs((dst_dir))
