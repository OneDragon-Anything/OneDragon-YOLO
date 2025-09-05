"""
图片相似度处理模块

该模块提供图片相似度计算和重复图片删除功能。
支持两种模式：
1. 跨文件夹比较：平衡各文件夹的图片数量
2. 文件夹内比较：删除每个文件夹内的重复图片
"""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import imagehash


class ImageSimilarityProcessor:
    """图片相似度处理器"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        初始化相似度处理器
        
        Args:
            similarity_threshold: 相似度阈值 (0-1)，越高越严格
        """
        self.similarity_threshold = similarity_threshold
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
    def get_image_files(self, folder_path: str) -> List[str]:
        """
        获取文件夹中的所有图片文件
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            图片文件路径列表
        """
        image_files = []
        
        if not os.path.exists(folder_path):
            return image_files
            
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in self.image_extensions:
                image_files.append(os.path.join(folder_path, file))
                
        return image_files
        
    def calculate_image_hash(self, image_path: str) -> Optional[imagehash.ImageHash]:
        """
        计算图片的感知哈希值
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            图片的感知哈希值，如果计算失败返回None
        """
        try:
            with Image.open(image_path) as img:
                # 使用感知哈希算法，对图片的小幅变化不敏感
                return imagehash.phash(img)
        except Exception as e:
            print(f"无法处理图片 {image_path}: {str(e)}")
            return None
            
    def calculate_similarity(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> float:
        """
        计算两个哈希值的相似度
        
        Args:
            hash1: 第一个图片的哈希值
            hash2: 第二个图片的哈希值
            
        Returns:
            相似度 (0-1)，1表示完全相同
        """
        hamming_distance = hash1 - hash2
        # 感知哈希通常是64位，最大汉明距离为64
        max_distance = 64
        similarity = 1.0 - (hamming_distance / max_distance)
        return similarity
        
    def get_folder_info(self, root_folder: str) -> List[Tuple[str, str, int]]:
        """
        获取根文件夹下所有子文件夹的信息
        
        Args:
            root_folder: 根文件夹路径
            
        Returns:
            [(文件夹名, 文件夹路径, 图片数量)] 的列表，按图片数量排序
        """
        folder_info = []
        
        if not os.path.exists(root_folder):
            return folder_info
            
        subfolders = [f for f in os.listdir(root_folder) 
                     if os.path.isdir(os.path.join(root_folder, f))]
        
        for folder in subfolders:
            folder_path = os.path.join(root_folder, folder)
            image_files = self.get_image_files(folder_path)
            folder_info.append((folder, folder_path, len(image_files)))
            
        # 按图片数量排序，从少到多
        folder_info.sort(key=lambda x: x[2])
        return folder_info
        
    def calculate_all_hashes(self, folder_info: List[Tuple[str, str, int]]) -> Dict[str, imagehash.ImageHash]:
        """
        计算所有图片的哈希值
        
        Args:
            folder_info: 文件夹信息列表
            
        Returns:
            {图片路径: 哈希值} 的字典
        """
        all_hashes = {}
        
        for folder_name, folder_path, image_count in folder_info:
            image_files = self.get_image_files(folder_path)
            for image_path in image_files:
                hash_value = self.calculate_image_hash(image_path)
                if hash_value is not None:
                    all_hashes[image_path] = hash_value
                    
        return all_hashes
        
    def process_cross_folder_similarity(self, root_folder: str, 
                                      progress_callback=None, 
                                      log_callback=None) -> Dict:
        """
        跨文件夹相似度处理
        
        从图片数量少的文件夹开始，与其他文件夹比较，
        删除其他文件夹中的相似图片，以平衡各文件夹的图片数量。
        
        Args:
            root_folder: 根文件夹路径
            progress_callback: 进度回调函数 (current, total, message)
            log_callback: 日志回调函数 (message)
            
        Returns:
            处理结果统计字典
        """
        result = {
            'deleted_files': 0,
            'processed_folders': 0,
            'total_comparisons': 0,
            'deleted_file_paths': []
        }
        
        # 获取文件夹信息
        folder_info = self.get_folder_info(root_folder)
        
        if len(folder_info) < 2:
            if log_callback:
                log_callback("需要至少2个子文件夹才能进行跨文件夹比较")
            return result
            
        if log_callback:
            log_callback(f"找到 {len(folder_info)} 个子文件夹")
            for folder, _, count in folder_info:
                log_callback(f"  {folder}: {count} 张图片")
                
        # 计算所有图片的哈希值
        all_hashes = self.calculate_all_hashes(folder_info)
        total_images = len(all_hashes)
        
        if progress_callback:
            progress_callback(total_images, total_images, "哈希值计算完成")
            
        # 跨文件夹相似度比较
        if log_callback:
            log_callback("开始跨文件夹相似度比较...")
            
        for i, (source_folder, source_path, _) in enumerate(folder_info[:-1]):
            source_images = [path for path in all_hashes.keys() 
                           if path.startswith(source_path)]
            
            for j in range(i + 1, len(folder_info)):
                target_folder, target_path, _ = folder_info[j]
                target_images = [path for path in all_hashes.keys() 
                               if path.startswith(target_path)]
                
                # 比较源文件夹和目标文件夹的图片
                for source_img in source_images:
                    if source_img not in all_hashes:
                        continue
                        
                    source_hash = all_hashes[source_img]
                    
                    for target_img in target_images[:]:  # 使用切片复制
                        if target_img not in all_hashes:
                            continue
                            
                        target_hash = all_hashes[target_img]
                        similarity = self.calculate_similarity(source_hash, target_hash)
                        result['total_comparisons'] += 1
                        
                        if similarity >= self.similarity_threshold:
                            # 删除目标文件夹中的相似图片
                            try:
                                os.remove(target_img)
                                del all_hashes[target_img]
                                target_images.remove(target_img)
                                result['deleted_files'] += 1
                                result['deleted_file_paths'].append(target_img)
                                
                                if log_callback:
                                    log_callback(
                                        f"删除相似图片: {os.path.basename(target_img)} "
                                        f"(相似度: {similarity:.3f})"
                                    )
                            except Exception as e:
                                if log_callback:
                                    log_callback(f"删除文件失败 {target_img}: {str(e)}")
                                    
        result['processed_folders'] = len(folder_info)
        return result
        
    def process_within_folder_similarity(self, root_folder: str, 
                                       progress_callback=None, 
                                       log_callback=None) -> Dict:
        """
        文件夹内相似度处理
        
        在每个子文件夹内部查找并删除相似的图片。
        
        Args:
            root_folder: 根文件夹路径
            progress_callback: 进度回调函数 (current, total, message)
            log_callback: 日志回调函数 (message)
            
        Returns:
            处理结果统计字典
        """
        result = {
            'deleted_files': 0,
            'processed_folders': 0,
            'total_comparisons': 0,
            'deleted_file_paths': []
        }
        
        # 获取所有子文件夹
        folder_info = self.get_folder_info(root_folder)
        
        for folder_name, folder_path, image_count in folder_info:
            if image_count < 2:
                continue
                
            if log_callback:
                log_callback(f"处理文件夹: {folder_name} ({image_count} 张图片)")
                
            image_files = self.get_image_files(folder_path)
            
            # 计算所有图片的哈希值
            hashes = {}
            for image_path in image_files:
                hash_value = self.calculate_image_hash(image_path)
                if hash_value is not None:
                    hashes[image_path] = hash_value
                    
            # 在文件夹内进行相似度比较
            image_paths = list(hashes.keys())
            i = 0
            while i < len(image_paths):
                j = i + 1
                while j < len(image_paths):
                    img1, img2 = image_paths[i], image_paths[j]

                    if img1 not in hashes or img2 not in hashes:
                        j += 1
                        continue

                    similarity = self.calculate_similarity(hashes[img1], hashes[img2])
                    result['total_comparisons'] += 1

                    if similarity >= self.similarity_threshold:
                        # 删除文件名较长的图片（通常是重复的）
                        to_delete = img2 if len(os.path.basename(img2)) > len(os.path.basename(img1)) else img1

                        try:
                            os.remove(to_delete)
                            del hashes[to_delete]
                            image_paths.remove(to_delete)
                            result['deleted_files'] += 1
                            result['deleted_file_paths'].append(to_delete)

                            if log_callback:
                                log_callback(
                                    f"删除相似图片: {os.path.basename(to_delete)} "
                                    f"(相似度: {similarity:.3f})"
                                )

                            # 如果删除的是img2，不需要增加j，因为列表已经缩短
                            # 如果删除的是img1，需要重新开始这一轮比较
                            if to_delete == img1:
                                break  # 跳出内层循环，重新开始外层循环
                        except Exception as e:
                            if log_callback:
                                log_callback(f"删除文件失败 {to_delete}: {str(e)}")
                            j += 1
                    else:
                        j += 1

                i += 1
                                
            result['processed_folders'] += 1
            
        return result
