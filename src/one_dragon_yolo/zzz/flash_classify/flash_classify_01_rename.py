import os

from tqdm import tqdm

from one_dragon_yolo.zzz.flash_classify import flash_classify_env


def main():
    project_dir = flash_classify_env.get_label_studio_project_dir()
    for sub_dir_name in os.listdir(project_dir):
        if sub_dir_name.startswith('.'):  # 忽略隐藏文件
            continue
        sub_dir = os.path.join(project_dir, sub_dir_name)
        for image_name in tqdm(os.listdir(sub_dir)):
            old_image_path = os.path.join(sub_dir, image_name)

            last_idx = image_name.rfind('_')
            if last_idx == -1:  # 根据创建时间命名
                file_stat = os.stat(old_image_path)
                timestamp = file_stat.st_ctime * 1000
                new_image_name = f'{timestamp:.0f}.png'
            else:
                new_image_name = image_name[last_idx + 1:]
            new_image_path = os.path.join(sub_dir, new_image_name)

            os.rename(old_image_path, new_image_path)

if __name__ == '__main__':
    main()