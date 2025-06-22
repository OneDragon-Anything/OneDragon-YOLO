import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics.data.split import autosplit

from one_dragon_yolo.devtools import ultralytics_utils, od_dataset_utils


class DataWrapper:

    def __init__(self, data_id: str, image_path: str, yolo_txt_path: str):
        self.data_id: str = data_id
        self.image_path: str = image_path
        self.yolo_txt_path: str = yolo_txt_path


def init_dataset_images_and_labels(
        dataset_name: str,
        data_list: list[DataWrapper],
        target_img_size: int = 2176
) -> bool:
    """
    初始化一个数据集的图片和标签

    原图是 1920*1080 会使用两张图片合并成正方形图片 同时将对应标签合并
    - 2176*2176 (2176=32*68)
    - 2208*2208 (2208=32*69)

    图片大小需要是32倍数 是因为 YOLO 模型需要5次下采样
    正方形是ultralytics默认的处理图片方式，合并后整张图信息更多，不会有很多空白区域

    Args:
        dataset_name: ultralytics数据集名称 在 ultralytics/datasets/{dataset_name}
        data_list: 原始数据
        target_img_size: 目标图片大小

    Returns:

    """
    if (target_img_size < 1080 * 2) or (target_img_size % 32 != 0):
        print('传入的图片大小不合法')
        return False

    # 删除已存在的数据集
    target_dataset_dir = ultralytics_utils.get_dataset_dir(dataset_name)
    shutil.rmtree(target_dataset_dir, ignore_errors=True)
    os.mkdir(target_dataset_dir)

    target_img_dir = ultralytics_utils.get_dataset_images_dir(dataset_name)
    os.mkdir(target_img_dir)

    target_label_dir = ultralytics_utils.get_dataset_labels_dir(dataset_name)
    os.mkdir(target_label_dir)

    total_cnt = len(data_list)

    for case1_idx in tqdm(range(total_cnt), desc='初始化数据集图片'):
        case2_idx = random.randint(0, total_cnt-1)
        
        case1 = data_list[case1_idx]
        case2 = data_list[case2_idx]

        img1 = cv2.imread(case1.image_path)
        label1_df = read_label_txt(case1.yolo_txt_path)

        img2 = cv2.imread(case2.image_path)
        label2_df = read_label_txt(case2.yolo_txt_path)

        height = img1.shape[0]
        width = img1.shape[1]
        radius = target_img_size

        save_img = np.full((radius, radius, 3), 114, dtype=np.uint8)
        save_img[0:height, 0:width, :] = img1
        save_img[height:height+height, 0:width, :] = img2

        label1_df['x'] *= width
        label1_df['y'] *= height
        label1_df['w'] *= width
        label1_df['h'] *= height

        label2_df['x'] *= width
        label2_df['y'] *= height
        label2_df['y'] += height
        label2_df['w'] *= width
        label2_df['h'] *= height

        save_label_df = pd.concat([label1_df, label2_df])
        save_label_df['x'] /= radius
        save_label_df['y'] /= radius
        save_label_df['w'] /= radius
        save_label_df['h'] /= radius

        save_img_path = os.path.join(target_img_dir, '%s-%s.png' % (case1.data_id, case2.data_id))
        cv2.imwrite(save_img_path, save_img)

        save_label_path = os.path.join(target_label_dir, '%s-%s.txt' % (case1.data_id, case2.data_id))
        save_label_df.to_csv(save_label_path, sep=' ', index=False, header=False)

    return True


def read_label_txt(txt_path) -> pd.DataFrame:
    """
    读取一个标签文件
    """
    return pd.read_csv(txt_path, sep=' ', header=None, encoding='utf-8', names=['idx', 'x', 'y', 'w', 'h'])


def init_dataset(
        project_dir: str,
        dataset_name: str,
        labels: list[str],
        target_img_size: int = 2176,
        split_weights=(0.9, 0.1, 0),
):
    # 读取图片和标签
    id_2_image = od_dataset_utils.get_yolo_data_image_path(project_dir)
    id_2_txt = od_dataset_utils.get_yolo_data_txt_path(project_dir)

    # 选取同时有图片和标注的id
    data_list: list[DataWrapper] = []
    for data_id in id_2_image.keys():
        if not data_id in id_2_txt:
            continue
        data_list.append(DataWrapper(data_id, id_2_image[data_id], id_2_txt[data_id]))

    # 初始化数据集
    init_dataset_images_and_labels(
        dataset_name=dataset_name,
        data_list=data_list,
        target_img_size=target_img_size
    )

    # 划分数据集
    target_dataset_dir = ultralytics_utils.get_dataset_dir(dataset_name)
    autosplit(path=ultralytics_utils.get_dataset_images_dir(dataset_name), weights=split_weights, annotated_only=True)
    if split_weights[1] == 0:
        train_txt_path = os.path.join(target_dataset_dir, 'autosplit_train.txt')
        val_txt_path = os.path.join(target_dataset_dir, 'autosplit_val.txt')
        shutil.copy(train_txt_path, val_txt_path)

    # 保存dataset.yaml
    with open(os.path.join(target_dataset_dir, 'dataset.yaml'), 'w', encoding='utf-8') as file:
        file.write('path: %s\n' % dataset_name)
        file.write('train: autosplit_train.txt\n')
        file.write('val: autosplit_val.txt\n')
        file.write('test: autosplit_test.txt\n')
        file.write('names:\n')
        for label_idx, label in enumerate(labels):
            file.write('  %d: %s\n' % (label_idx, label))
