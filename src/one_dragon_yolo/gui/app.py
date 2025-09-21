import sys
import os
import configparser
from PySide6.QtWidgets import QApplication

from qfluentwidgets import (setTheme, Theme, FluentWindow, FluentIcon)

from one_dragon_yolo.gui.image_similarity_tab import ImageSimilarityTab
from one_dragon_yolo.gui.image_classification_tab import ImageClassificationTab
from one_dragon_yolo.gui.image_validation_tab import ImageValidationTab


class ImageClassifierWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        # 初始化配置文件路径
        self.config_dir = os.path.join(os.getcwd(), ".conf")
        self.config_file = os.path.join(self.config_dir, "classifier_gui.ini")
        self.config = configparser.ConfigParser()
        
        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 加载配置
        self.load_config()
        
        self.setWindowTitle("图像处理工具")
        self.setGeometry(100, 100, 850, 800)


        # --- Image Classification Tab ---
        self.classification_tab = ImageClassificationTab()
        self.classification_tab.setObjectName('classification_interface')
        self.addSubInterface(self.classification_tab, FluentIcon.PHOTO, '图像分类')

        # --- Image Validation Tab ---
        self.validation_tab = ImageValidationTab()
        self.validation_tab.setObjectName('validation_interface')
        self.addSubInterface(self.validation_tab, FluentIcon.CERTIFICATE, '分类校验')

        # --- Image Similarity Tab ---
        self.similarity_tab = ImageSimilarityTab()
        self.similarity_tab.setObjectName('similarity_interface')
        self.addSubInterface(self.similarity_tab, FluentIcon.DELETE, '相似度删除')

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file, encoding='utf-8')
        else:
            # 如果配置文件不存在，创建默认配置
            if not self.config.has_section('Paths'):
                self.config.add_section('Paths')
            self.config['Paths'] = {
                'last_model_path': '',
                'last_image_dir': '',
                'last_output_dir': ''
            }
            self.save_config()

    def save_config(self):
        """保存配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            self.config.write(f)

    def get_config_value(self, key, default=''):
        """获取配置值"""
        try:
            return self.config.get('Paths', key, fallback=default)
        except:
            return default

    def set_config_value(self, key, value):
        """设置配置值"""
        if not self.config.has_section('Paths'):
            self.config.add_section('Paths')
        self.config.set('Paths', key, value)
        self.save_config()





if __name__ == "__main__":
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    window = ImageClassifierWindow()
    window.show()
    sys.exit(app.exec())