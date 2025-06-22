import json
import os

from tqdm import tqdm


class DataWrapper:

    def __init__(
            self,
            data_id: str,
            image_path: str,
            yolo_txt_path: str,
            x_json_path: str,
    ):
        self.data_id = data_id  # 数据唯一标识 通常是文件名不含后缀部分
        self.image_path = image_path  # 数据对应的图片路径
        self.yolo_txt_path = yolo_txt_path  # 数据对应的txt格式的YOLO结果路径
        self.x_json_path = x_json_path  # 数据对应的X-AnyLabeling json格式的标注结果路径


def get_image_raw_dir(project_dir: str) -> str:
    """
    获取一个数据集项目下的图片根目录
    """
    return os.path.join(project_dir, 'raw')


def get_yolo_txt_dir(project_dir: str) -> str:
    """
    获取一个数据集项目下的YOLO txt格式的标签根目录
    """
    return os.path.join(project_dir, 'yolo')


def get_x_json_dir(project_dir: str) -> str:
    """
    获取一个数据集项目下的X-AnyLabeling json格式的标注结果根目录
    """
    return os.path.join(project_dir, 'X-AnyLabeling', 'annotation')


def get_project_data_list(
        project_dir: str,
) -> list[DataWrapper]:
    """
    获取一个数据集项目下的所有数据
    """
    data_list = []
    raw_dir = get_image_raw_dir(project_dir)
    yolo_txt_dir = get_yolo_txt_dir(project_dir)
    x_json_dir = get_x_json_dir(project_dir)
    for sub_dir_name in tqdm(os.listdir(raw_dir), desc='读取已有标签文件夹'):
        sub_dir_path = os.path.join(raw_dir, sub_dir_name)
        if not os.path.isdir(sub_dir_path):
            continue

        for file_name in tqdm(os.listdir(sub_dir_path), desc='读取已有标签'):
            if not file_name.endswith('.png'):
                continue
            data_id = file_name[:-4]
            image_path = os.path.join(sub_dir_path, file_name)
            yolo_txt_path = os.path.join(yolo_txt_dir, f'{data_id}.txt')
            x_json_path = os.path.join(x_json_dir, f'{data_id}.json')

            data_list.append(
                DataWrapper(
                    data_id=data_id,
                    image_path=image_path,
                    yolo_txt_path=yolo_txt_path,
                    x_json_path=x_json_path,
                )
            )

    return data_list


def empty_x_data(
        image_path: str,
        image_width: int = 1920,
        image_height: int = 1080,
) -> dict:
    """
    创建一个空的 X-AnyLabeling 的 json 格式数据
    """
    return {
        "version": "3.0.3",
        "flags": {},
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": "",
        "shapes": [],
    }


def yolo_2_x(
        yolo: list[float],
        labels: list[str],
        image_width: int = 1920,
        image_height: int = 1080,
) -> dict:
    """
    将一行 yolo 数据转成 X-AnyLabeling 的 一个标注
    :param yolo: yolo数据 [cls, cx, cy, w, h]
    :param labels: 类别名称列表
    :param image_width: 图片宽度
    :param image_height: 图片高度
    """
    cx = yolo[1] * image_width
    cy = yolo[2] * image_height
    w = yolo[3] * image_width
    h = yolo[4] * image_height

    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    return {
        "label": labels[int(yolo[0])],
        "score": None,
        "points": [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": []
    }


def convert_yolo_2_x(
        yolo_txt_dir: str,
        x_json_dir: str,
        labels: list[str],
        image_width: int = 1920,
        image_height: int = 1080,
) -> None:
    """
    遍历文件夹
    将 txt格式的YOLO结果 转成 X-AnyLabeling 的 json 格式

    ```txt
    0 0.773914433084428 0.273859746754169 0.026951028034091 0.046918287873268
    ```

    ```json
    {
      "version": "3.0.3",
      "flags": {},
      "shapes": [
        {
          "label": "0000-感叹号",
          "score": null,
          "points": [
            [
              1250.2439024390244,
              268.0487804878049
            ],
            [
              1303.90243902439,
              268.0487804878049
            ],
            [
              1303.90243902439,
              318.0487804878049
            ],
            [
              1250.2439024390244,
              318.0487804878049
            ]
          ],
          "group_id": null,
          "description": "",
          "difficult": false,
          "shape_type": "rectangle",
          "flags": {},
          "attributes": {},
          "kie_linking": []
        }
      ],
      "imagePath": "0000-感叹号-0001.png",
      "imageData": null,
      "imageHeight": 1080,
      "imageWidth": 1920,
      "description": ""
    }
    ```
    """
    for txt_name in tqdm(os.listdir(yolo_txt_dir)):
        txt_path = os.path.join(yolo_txt_dir, txt_name)

        yolo_arr = []

        with open(txt_path, 'r', encoding='utf-8') as f:
            txt_lines = f.readlines()
            for txt_line in txt_lines:
                txt_line = txt_line.strip()
                if not txt_line:
                    continue
                yolo_arr.append([float(x) for x in txt_line.split(' ')])

        image_name = txt_name.replace('.txt', '.png')
        json_name = txt_name.replace('.txt', '.json')
        json_path = os.path.join(x_json_dir, json_name)
        json_data = empty_x_data(image_name, image_width, image_height)

        for yolo in yolo_arr:
            json_data['shapes'].append(yolo_2_x(yolo, labels, image_width, image_height))

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
