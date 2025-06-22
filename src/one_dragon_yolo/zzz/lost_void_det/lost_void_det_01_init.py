from one_dragon_yolo.devtools import label_studio_utils
from one_dragon_yolo.zzz.lost_void_det import lost_void_det_env

# 初始化文件夹 以及输出label_studio的标签
if __name__ == '__main__':
    label_df = lost_void_det_env.get_label_df()
    project_dir = lost_void_det_env.get_dataset_project_dir()

    label_studio_utils.create_sub_dir_in_raw(
        project_dir=project_dir,
        label_df=label_df,
        label_col='label',
        class_col='entry_name'
    )

    label_studio_utils.print_labeling_interface(
        label_df=label_df,
        label_col='label',
        class_col='entry_name'
    )