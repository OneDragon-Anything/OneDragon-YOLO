import os
from typing import List

import pandas as pd

from one_dragon_yolo.devtools import od_dataset_utils
from one_dragon_yolo.devtools import os_utils


def get_dataset_project_dir() -> str:
    """
    获取原始数据集项目的根目录
    """
    return od_dataset_utils.get_dataset_project_dir('ZZZ-LostVoidDet-Dataset')


def get_label_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(
        os_utils.get_path_under_work_dir('labels', 'zzz'),
        'lost_void_det.csv'
    ))


def get_labels_with_name() -> List[str]:
    label_df = get_label_df()
    result = []
    for index, row in label_df.iterrows():
        result.append('%04d-%s' % (row['label'], row['entry_name']))
    return result