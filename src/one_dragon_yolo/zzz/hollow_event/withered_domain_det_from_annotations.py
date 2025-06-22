from one_dragon_yolo.devtools import label_studio_utils

if __name__ == '__main__':
    project_dir = label_studio_utils.get_label_studio_project_dir('zzz', 'hollow_event')
    label_studio_utils.generate_tasks_from_annotations(
        project_dir,
        old_img_path_prefix='zzz\\hollow_event\\raw',
        new_img_path_prefix='ZZZ-WitheredDomainDet-Dataset\\raw',
    )