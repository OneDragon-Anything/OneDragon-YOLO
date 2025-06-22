import math

from ultralytics import YOLO

from one_dragon_yolo.devtools import ultralytics_utils
from one_dragon_yolo.devtools import yolo_dataset_utils
from one_dragon_yolo.zzz.lost_void_det import lost_void_det_env

# 训练
if __name__ == "__main__":
    ultralytics_utils.init_ultralytics_settings()

    # origin_img_size = 32 * 68
    # img_size_div = 2
    dataset_img_size = 32 * 69  # 处理后的数据集的图片大小 正方形 YOLO需要图片大小是32的倍数 模型有5次下采样
    img_size_div = 3
    train_img_size = dataset_img_size // img_size_div   # 训练时的使用的图片大小

    export_width = dataset_img_size // img_size_div  # 图片宽度比高度大 使用宽度作为基准
    export_height = math.ceil((export_width // 16 * 9) * 1.0 / 32) * 32  # 高度按16:9调整
    export_img_size = (export_height, export_width)  # 由于训练时候没有开启缩放，使用训练的尺寸效果会更好

    train_dataset_name = f'zzz_lost_void_det_{dataset_img_size}'

    # pretrained_model_name = 'yolo11n'
    pretrained_model_name = 'yolov8n'
    train_name = f'{pretrained_model_name}-{train_img_size}'  # 训练过程数据 runs中的文件夹名称

    print(train_dataset_name, train_name, export_img_size)

    yolo_dataset_utils.init_dataset(
        project_dir=lost_void_det_env.get_dataset_project_dir(),
        dataset_name=train_dataset_name,
        labels=lost_void_det_env.get_labels_with_name(),
        target_img_size=dataset_img_size,
        split_weights=(0.9, 0.1, 0),
    )

    model = YOLO(ultralytics_utils.get_base_model_path(f'{pretrained_model_name}.pt'))

    model.train(
        data=ultralytics_utils.get_dataset_yaml_path(train_dataset_name),  # 数据集配置文件的位置
        project=ultralytics_utils.get_dataset_model_dir(train_dataset_name),  # 训练模型的数据（包括模型文件）的自动保存位置
        name=train_name,
        imgsz=train_img_size,
        epochs=200,
        save_period=100,
        batch=-1,  # 根据可使用内存 自动判断batch_size
        val=False,  # 关闭验证
        exist_ok=True,
        scale=0,  # 不需要缩放处理
        flipud=0, fliplr=0,  # 不需要翻转对称
        erasing=0,  # 不需要消除了 图片下方空白比较多
        hsv_h=0, hsv_s=0, hsv_v=0,  # 关闭色彩调节
        mosaic=0,  # 不需要拼接 使用原装大小
    )

    ultralytics_utils.export_model(
        dataset_name=train_dataset_name,
        train_name=train_name,
        imgsz=export_img_size
    )
