from one_dragon_yolo.devtools import label_studio_utils


def get_label_studio_project_dir() -> str:
    """
    获取label studio项目目录
    """
    return label_studio_utils.get_label_studio_project_dir('ZZZ-FlashClassify-Dataset')
