from one_dragon_yolo.devtools import label_studio_utils

if __name__ == '__main__':
    project_dir = label_studio_utils.get_label_studio_project_dir('sr', 'object_detect')
    label_studio_utils.generate_tasks_from_annotations(
        project_dir,
        old_img_path_prefix='sr\\object_detect\\raw',
        new_img_path_prefix='SR-ObjectDet-Dataset\\raw',
    )