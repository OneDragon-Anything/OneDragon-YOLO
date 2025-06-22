import cv2
import imagehash
import numpy as np
from PIL import Image


def calculate_phash_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    计算两张图片的感知哈希 (PHash) 相似度。
    通过计算哈希值的汉明距离来判断相似性。
    相似度值介于 0 到 1 之间，1 表示完全相同（汉明距离为 0）。

    Args:
        image1 (np.ndarray): 第一张 OpenCV 格式的图片 (MatLike)。
        image2 (np.ndarray): 第二张 OpenCV 格式的图片 (MatLike)。

    Returns:
        float: 基于 PHash 汉明距离的相似度值。
               相似度 = 1 - (汉明距离 / 哈希位长)。
    """
    # 将 OpenCV 图像转换为 PIL 图像，因为 imagehash 库更常用 PIL 图像
    # 注意：cv2.cvtColor 只是示例，如果图像已经是灰度图则不需要
    image1_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2_pil = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    # 计算感知哈希
    hash1 = imagehash.phash(image1_pil)
    hash2 = imagehash.phash(image2_pil)

    # 计算汉明距离
    hamming_distance = hash1 - hash2
    
    # 哈希的位长 (通常是 64 位，即 8*8)
    # 对于 imagehash.phash，默认是 8x8 的哈希，所以位长是 64
    hash_bits = hash1.hash.size * hash1.hash.itemsize * 8 # itemsize for byte length, *8 for bits

    # 计算相似度 (1 表示完全相同，0 表示完全不同)
    similarity = 1 - (hamming_distance / hash_bits)
    
    return similarity