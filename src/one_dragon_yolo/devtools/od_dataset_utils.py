import os

from one_dragon_yolo.devtools import env_utils


def get_dataset_project_dir(project: str) -> str:
    """
    数据集项目的根目录
    :param project: 数据集项目的项目名称
    """
    return os.path.join(env_utils.DATASET_PARENT_DIR, project)


def get_yolo_raw_dir(project_dir: str) -> str:
    """
    获取YOLO数据集项目下 原图文件夹的位置
    :param project_dir: 数据集项目位置
    """
    return os.path.join(project_dir, 'raw')


def get_yolo_txt_dir(project_dir: str) -> str:
    """
    获取YOLO数据集项目下 标注文件文件夹的位置
    :param project_dir: 数据集项目位置
    """
    return os.path.join(project_dir, 'yolo')


def get_yolo_x_json_dir(project_dir: str) -> str:
    """
    获取YOLO数据集项目下 X-AnyLabeling json格式的标注结果根目录

    Args:
        project_dir: 数据集项目位置

    Returns:
        x_json_dir: X-AnyLabeling json格式的标注结果根目录
    """
    return os.path.join(project_dir, 'X-AnyLabeling', 'annotation')


def get_yolo_data_image_path(
        project_dir: str
) -> dict[str, str]:
    """
    获取数据集中 数据图片的路径

    Args:
        project_dir: 数据集项目位置

    Returns:
        dict[str, str]: key=数据ID value=数据图片路径

    """
    result = {}
    raw_dir = get_yolo_raw_dir(project_dir)
    for sub_dir_name in os.listdir(raw_dir):
        sub_dir_path = os.path.join(raw_dir, sub_dir_name)
        if not os.path.isdir(sub_dir_path):
            continue
        for file_name in os.listdir(sub_dir_path):
            if not file_name.endswith('.png'):
                continue
            data_id = file_name[:-4]
            image_path = os.path.join(sub_dir_path, file_name)
            result[data_id] = image_path

    return result


def get_yolo_data_txt_path(
        project_dir: str,
) -> dict[str, str]:
    """
    获取数据集中 YOLO标注的txt文件路径
    Args:
        project_dir: 数据集项目根目录

    Returns:
        dict[str, str]: key=数据ID value=标注txt路径
    """
    result = {}
    txt_dir = get_yolo_txt_dir(project_dir)
    for file_name in os.listdir(txt_dir):
        if not file_name.endswith('.txt'):
            continue
        data_id = file_name[:-4]
        result[data_id] = os.path.join(txt_dir, file_name)

    return result


def rename_file_in_yolo_project(project_dir: str) -> None:
    """
    对原图的文件夹下的图片进行重命名
    如果有对应的 yolo 标签，也一起重命名
    :param project_dir: 项目目录
    """
    raw_dir = get_yolo_raw_dir(project_dir)
    data_2_yolo = get_yolo_data_txt_path(project_dir)

    for sub_dir_name in os.listdir(raw_dir):
        sub_dir = os.path.join(raw_dir, sub_dir_name)
        if not os.path.isdir(sub_dir):
            continue

        max_idx: int = 0
        for image_name in os.listdir(sub_dir):
            if not image_name.endswith('.png'):
                continue
            if image_name.startswith(sub_dir_name):
                idx = int(image_name[-8:-4])
                max_idx = max(idx, max_idx)
        max_idx += 1

        for image_name in os.listdir(sub_dir):
            if not image_name.endswith('.png'):
                continue

            if image_name.startswith(sub_dir_name):
                continue

            old_data_id = image_name[:-4]
            new_data_id = '%s-%04d' % (sub_dir_name, max_idx)
            print(f'{old_data_id} -> {new_data_id}')

            old_image_path = os.path.join(sub_dir, image_name)
            new_image_path = os.path.join(sub_dir, f'{new_data_id}.png')
            os.rename(old_image_path, new_image_path)

            # 如果有yolo标注文件 也重命名
            if old_data_id in data_2_yolo:
                old_txt_path = data_2_yolo[old_data_id]
                new_txt_path = os.path.join(os.path.dirname(old_txt_path), f'{new_data_id}.txt')
                os.rename(old_txt_path, new_txt_path)

            max_idx += 1


def convert_yolo_2_x(project_name: str) -> None:
    """
    将数据集项目中的YOLO txt文件 转换同步到 X-AnyLabeling 的json格式
    Args:
        project_name: 项目名称

    Returns:

    """
    project_dir = get_dataset_project_dir()
    labels = lost_void_det_env.get_labels_with_name()

    x_dir = os.path.join(
        lost_void_det_env.get_dataset_project_dir(),
        'X-AnyLabeling',
        'annotation'
    )

    x_anylabeling_utils.convert_yolo_2_x(
        yolo_txt_dir=yolo_txt_dir,
        x_json_dir=x_dir,
        labels=labels,
    )