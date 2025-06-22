import os
import random
import shutil

from tqdm import tqdm


def split_dataset(
        raw_dataset_dir: str,
        split_dataset_dir: str,
        split_weights=(0.9, 0.1)
):
    """
    分隔数据集 每个分类都保持相同的比例
    """
    new_train_dir_path = os.path.join(split_dataset_dir, 'train')
    shutil.rmtree(new_train_dir_path, ignore_errors=True)
    os.mkdir(new_train_dir_path)

    new_val_dir_path = os.path.join(split_dataset_dir, 'val')
    shutil.rmtree(new_val_dir_path, ignore_errors=True)
    os.mkdir(new_val_dir_path)

    for class_dir_name in tqdm(os.listdir(raw_dataset_dir), desc='Copying by class'):
        if class_dir_name[0] == '.':
            # 忽略隐藏文件夹 可能是 .git 之类的
            continue
        old_class_dir_path = os.path.join(raw_dataset_dir, class_dir_name)
        old_class_image_name_list = os.listdir(old_class_dir_path)

        # 随机打乱数组
        random.shuffle(old_class_image_name_list)

        # 拆分
        train_cnt = int(len(old_class_image_name_list) * split_weights[0])
        old_class_train_image_name_list = old_class_image_name_list[:train_cnt]
        old_class_val_image_name_list = old_class_image_name_list[train_cnt:]

        # train目录下 创建新的分类文件夹 并复制图片
        train_new_class_dir_path = os.path.join(new_train_dir_path, class_dir_name)
        shutil.rmtree(train_new_class_dir_path, ignore_errors=True)
        os.mkdir(train_new_class_dir_path)
        for image_name in tqdm(old_class_train_image_name_list, desc='Copying to train'):
            old_image_path = os.path.join(old_class_dir_path, image_name)
            new_image_path = os.path.join(train_new_class_dir_path, image_name)
            shutil.copyfile(old_image_path, new_image_path)

        # val目录下 创建新的分类文件夹 并复制图片
        val_new_class_dir_path = os.path.join(new_val_dir_path, class_dir_name)
        shutil.rmtree(val_new_class_dir_path, ignore_errors=True)
        os.mkdir(val_new_class_dir_path)
        for image_name in tqdm(old_class_val_image_name_list, desc='Copying to val'):
            old_image_path = os.path.join(old_class_dir_path, image_name)
            new_image_path = os.path.join(val_new_class_dir_path, image_name)
            shutil.copyfile(old_image_path, new_image_path)