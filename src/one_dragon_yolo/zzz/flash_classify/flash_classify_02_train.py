import torchvision.transforms as T
from ultralytics import YOLO
from ultralytics.data import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer
from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.utils import DEFAULT_CFG

from one_dragon_yolo.devtools import ultralytics_utils, classify_dataset_utils
from one_dragon_yolo.image_modules.square_pad import SquarePad
from one_dragon_yolo.zzz.flash_classify import flash_classify_env


class FlashClassifyDataset(ClassificationDataset):

    def __init__(self, root, args, augment=False, prefix=""):
        ClassificationDataset.__init__(self, root=root, args=args, augment=augment, prefix=prefix)

        # 覆盖原有的数据增强
        # 不缩放 不翻转 不变色 不消除
        self.torch_transforms = T.Compose(
            [
                SquarePad(after_size=args.imgsz),  # 补成正方形 适配后续导出onnx
                T.ToTensor(),  #  如果原始图像的像素值在 [0, 255] 之间 (例如，uint8 类型)，ToTensor() 会自动将其缩放到 [0.0, 1.0] 范围，并将数据类型转换为 float32
            ]
        )


class FlashClassifyTrainer(ClassificationTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        ClassificationTrainer.__init__(self, cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        return FlashClassifyDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class FlashClassifyValidator(ClassificationValidator):

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        ClassificationValidator.__init__(
            self,
            dataloader=dataloader,
            save_dir=save_dir,
            args=args,
            _callbacks=_callbacks
        )

    def build_dataset(self, img_path):
        return FlashClassifyDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)


if __name__ == '__main__':
    ultralytics_utils.init_ultralytics_settings()
    model = YOLO(ultralytics_utils.get_base_model_path('yolov8n-cls.pt'))

    dataset_name = 'zzz_flash_raw'
    train_name = 'train'
    export_img_size = 640
    model_name = 'best'

    # 数据集原始位置 第一层应该是各个类别的文件夹
    raw_dataset_dir = flash_classify_env.get_label_studio_project_dir()
    train_dataset_dir = ultralytics_utils.get_dataset_dir(dataset_name)
    classify_dataset_utils.split_dataset(
        raw_dataset_dir=raw_dataset_dir,
        split_dataset_dir=train_dataset_dir,
    )

    model.train(
        data=train_dataset_dir,
        trainer=FlashClassifyTrainer,
        project=ultralytics_utils.get_dataset_model_dir(dataset_name),  # 训练模型的数据（包括模型文件）的自动保存位置
        epochs=200,
        imgsz=640,
        batch=-1,  # 根据可使用内存 自动判断batch_size
        val=False,  # 关闭验证
        exist_ok=True,
    )

    model.val(
        data=train_dataset_dir,
        validator=FlashClassifyValidator,
        imgsz=640,
        batch=10,
    )

    ultralytics_utils.export_cls_model(
        raw_dataset_dir=raw_dataset_dir,
        dataset_name=dataset_name,
        train_name=train_name,
        imgsz=export_img_size,
        model_name=model_name,
        save_name=f'{train_name}-{model_name}'
    )