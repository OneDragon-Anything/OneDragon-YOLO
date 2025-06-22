# OneDragon-YOLO

用于一条龙脚本的YOLO项目

## 开发环境说明

- Python版本 = 3.11.9
- CUDA版本 = 12.6 Windows x86_64 Version 11
- 先安装 cuda 版的 pytorch，可参考[官网](https://pytorch.org/get-started/locally/)
  ```shell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
- 再安装 ultralytics 和 其他依赖
  ```shell
  pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple`
  ```

### 为什么不使用 uv

因为 uv 安装 torch 包的时候，依赖 `markupsafe` 只有 python 3.13 版本，而 ultralytics 最高只支持 3.12


## 使用

### 数据集

1. 在你的电脑上找一个位置，创建一个文件夹，例如 `OneDragon-Dataset`
2. 进入这个文件夹后，克隆所需的数据集
    - [绝区零-迷失之地](https://www.modelscope.cn/datasets/DoctorReid/ZZZ-LostVoidDet-Dataset)
3. 复制 `.env.example` 重命名成 `.env`，修改
    - `DATASET_PARENT_DIR`: `OneDragon-Dataset`的目录
4. 根据你要处理的数据集，进入到不同的package中查看运行
    - 绝区零-迷失之地: `src/one_dragon_yolo/zzz/lost_void_det`