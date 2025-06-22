from ultralytics import YOLO

from one_dragon_yolo.devtools import ultralytics_utils
from one_dragon_yolo.zzz.flash_classify import flash_classify_env
from one_dragon_yolo.zzz.flash_classify.flash_classify_03_train import FlashClassifyValidator

if __name__ == '__main__':
    dataset_name = 'zzz_flash_raw'
    train_name = 'train'
    export_img_size = 640
    model_name = 'best'

    pt_model_path = ultralytics_utils.get_train_model_path(dataset_name, train_name, model_name, model_type='pt')
    model = YOLO(pt_model_path)

    # 数据集原始位置 第一层应该是各个类别的文件夹
    raw_dataset_dir = flash_classify_env.get_label_studio_project_dir()

    model.val(
        data=raw_dataset_dir,
        validator=FlashClassifyValidator,
        imgsz=640,
        batch=10,
    )