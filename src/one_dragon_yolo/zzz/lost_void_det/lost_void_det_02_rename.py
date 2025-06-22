from one_dragon_yolo.devtools import od_dataset_utils, x_anylabeling_utils
from one_dragon_yolo.zzz.lost_void_det import lost_void_det_env

# 往数据集添加新的图片后 按规定格式对图片进行重命名
# 同时会将YOLO结果(如果有的话)同步到 X-AnyLabeling 的json格式
if __name__ == '__main__':
    project_dir = lost_void_det_env.get_dataset_project_dir()

    od_dataset_utils.rename_file_in_yolo_project(
        project_dir=project_dir
    )

    yolo_txt_dir = od_dataset_utils.get_yolo_txt_dir(project_dir)
    x_json_dir = od_dataset_utils.get_yolo_x_json_dir(project_dir)
    labels = lost_void_det_env.get_labels_with_name()

    x_anylabeling_utils.convert_yolo_2_x(
        yolo_txt_dir=yolo_txt_dir,
        x_json_dir=x_json_dir,
        labels=labels,
    )