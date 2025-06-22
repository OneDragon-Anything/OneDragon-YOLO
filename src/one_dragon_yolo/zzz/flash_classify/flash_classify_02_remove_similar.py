import os

from one_dragon_yolo.devtools import common_dataset_utils
from one_dragon_yolo.zzz.flash_classify import flash_classify_env


def main() -> None:
    project_dir = flash_classify_env.get_label_studio_project_dir()
    for sub_dir_name in os.listdir(project_dir):
        sub_dir = os.path.join(project_dir, sub_dir_name)
        common_dataset_utils.remove_similar_image(
            sub_dir,
            similarity_threshold=0.99
        )


if __name__ == '__main__':
    main()