import sys
import os
import shutil
from PySide6.QtCore import Qt, QSize, QTimer, QSettings
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel

from qfluentwidgets import (setTheme, Theme, FluentWindow, PushButton, PrimaryPushButton,
                            BodyLabel, ComboBox, MessageBox, InfoBar, InfoBarPosition, FluentIcon,
                            DoubleSpinBox, SwitchButton, SubtitleLabel, LineEdit)
from ultralytics import YOLO
from PIL import Image
import configparser

from one_dragon_yolo.gui.image_similarity_tab import ImageSimilarityTab


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

        # --- Main Widget and Layout ---
        self.main_widget = QWidget()
        self.main_widget.setObjectName('main_interface')
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        self.addSubInterface(self.main_widget, FluentIcon.PHOTO, '图像分类')

        # --- Image Similarity Tab ---
        self.similarity_tab = ImageSimilarityTab()
        self.similarity_tab.setObjectName('similarity_interface')
        self.addSubInterface(self.similarity_tab, FluentIcon.DELETE, '相似度删除')

        # --- UI Elements ---

        # Model selection
        model_layout = QHBoxLayout()
        self.btn_select_model = PushButton("选择模型文件")
        self.model_path_edit = LineEdit()
        self.model_path_edit.setPlaceholderText("请先选择一个.pt模型文件")
        self.model_path_edit.setReadOnly(True)
        model_layout.addWidget(self.btn_select_model)
        model_layout.addWidget(self.model_path_edit, 1)
        self.main_layout.addLayout(model_layout)

        # Top buttons
        top_button_layout = QHBoxLayout()
        self.btn_open_dir = PushButton("选择图片文件夹")
        self.btn_output_dir = PushButton("选择输出文件夹")
        top_button_layout.addWidget(self.btn_open_dir)
        top_button_layout.addWidget(self.btn_output_dir)
        self.main_layout.addLayout(top_button_layout)

        # Image display
        self.image_label = QLabel("请先加载模型，然后选择图片文件夹")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #2C2C2C; border-radius: 5px;")
        self.main_layout.addWidget(self.image_label, 1)

        self.lbl_image_path = BodyLabel("图片路径将显示在这里")
        self.lbl_image_path.setWordWrap(True)
        self.main_layout.addWidget(self.lbl_image_path)

        # Action buttons
        action_layout = QHBoxLayout()
        self.lbl_prediction = BodyLabel("预测分类: N/A")
        self.lbl_prediction.setObjectName('lbl_prediction')
        self.btn_next = PrimaryPushButton("下一张")
        self.btn_accept = PushButton("接收当前预测分类")
        self.btn_delete = PushButton("删除当前图片")
        action_layout.addWidget(self.lbl_prediction)
        action_layout.addStretch()
        action_layout.addWidget(self.btn_next)
        action_layout.addWidget(self.btn_accept)
        action_layout.addWidget(self.btn_delete)
        self.main_layout.addLayout(action_layout)

        # Manual classification
        manual_layout = QHBoxLayout()
        self.combo_manual_class = ComboBox()
        self.combo_manual_class.setPlaceholderText("手动选择分类")
        self.btn_manual_move = PushButton("移动到所选分类")
        manual_layout.addWidget(self.combo_manual_class)
        manual_layout.addWidget(self.btn_manual_move)
        self.main_layout.addLayout(manual_layout)

        # Auto-delete settings
        self.main_layout.addWidget(SubtitleLabel('自动删除设置'))
        auto_delete_layout = QHBoxLayout()
        self.threshold_spinbox = DoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setSingleStep(0.01)
        self.threshold_spinbox.setValue(1.0)
        self.threshold_spinbox.setPrefix("置信度阈值: ")
        self.auto_delete_switch = SwitchButton('启用自动删除')
        auto_delete_layout.addWidget(self.threshold_spinbox)
        auto_delete_layout.addWidget(self.auto_delete_switch)
        auto_delete_layout.addStretch()
        self.main_layout.addLayout(auto_delete_layout)

        # --- State Variables ---
        self.image_dir = ""
        self.output_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self.current_image_path = None
        self.model = None
        self.class_names = {}
        self.predicted_class = None

        # --- Connections ---
        self.btn_select_model.clicked.connect(self.select_and_load_model)
        self.btn_open_dir.clicked.connect(self.open_image_dir)
        self.btn_output_dir.clicked.connect(self.select_output_dir)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_accept.clicked.connect(self.accept_prediction)
        self.btn_delete.clicked.connect(self.delete_image)
        self.btn_manual_move.clicked.connect(self.manual_move)

        # --- Initial State ---
        self._update_button_states()

    def show_info(self, title, content, success=True):
        InfoBar.success(title, content, duration=3000, position=InfoBarPosition.TOP, parent=self) if success else InfoBar.error(title, content, duration=5000, position=InfoBarPosition.TOP, parent=self)

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

    def select_and_load_model(self):
        # 从配置中获取上次选择的模型路径作为默认路径
        last_model_path = self.get_config_value("last_model_path")
        last_model_dir = os.path.dirname(last_model_path) if os.path.exists(last_model_path) else ""
        
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", last_model_dir, "PyTorch Models (*.pt)"
        )
        if not model_path:
            return
        # 保存选择的模型路径
        self.set_config_value("last_model_path", model_path)
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                self.show_info("错误", f"模型文件不存在: {model_path}", success=False)
                return
            
            self.model = YOLO(model_path)
            # 从模型中获取分类名称
            self.class_names = self.model.names if isinstance(self.model.names, dict) else {i: name for i, name in enumerate(self.model.names)}
            
            self.combo_manual_class.clear()
            # 添加分类到下拉框
            self.combo_manual_class.addItems(list(self.class_names.values()))
            
            self.model_path_edit.setText(model_path)
            self.show_info("成功", f"模型加载成功，分类: {list(self.class_names.values())}")
        except Exception as e:
            self.model = None
            self.class_names = {}
            self.model_path_edit.clear()
            self.show_info("模型加载失败", str(e), success=False)
        
        self._update_button_states()

    def open_image_dir(self):
        # 从配置中获取上次选择的图片目录作为默认路径
        last_image_dir = self.get_config_value("last_image_dir")
        if not os.path.exists(last_image_dir):
            last_image_dir = ""
            
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", last_image_dir)
        if not dir_path:
            return

        # 保存选择的图片目录
        self.set_config_value("last_image_dir", dir_path)
        
        self.image_dir = dir_path
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not self.image_files:
            self.show_info("提示", "文件夹中没有找到图片", success=False)
            self.image_dir = ""
            return

        self.current_image_index = -1
        self.next_image()

    def select_output_dir(self):
        # 从配置中获取上次选择的输出目录作为默认路径
        last_output_dir = self.get_config_value("last_output_dir")
        if not os.path.exists(last_output_dir):
            last_output_dir = ""
            
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", last_output_dir)
        if not dir_path:
            return

        # 保存选择的输出目录
        self.set_config_value("last_output_dir", dir_path)
        
        self.output_dir = dir_path
        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            # 为每个分类创建子目录
            for class_name in self.class_names.values():
                os.makedirs(os.path.join(self.output_dir, class_name), exist_ok=True)
            self.show_info("成功", "输出文件夹设置成功，并已创建分类子文件夹。")
        except Exception as e:
            self.show_info("错误", f"创建分类文件夹失败: {e}", success=False)
            self.output_dir = ""
        self._update_button_states()

    def _update_button_states(self):
        has_model = self.model is not None
        has_image_dir = self.image_dir != ""
        has_current_image = self.current_image_path is not None and os.path.exists(self.current_image_path)
        has_output_dir = self.output_dir != ""
        has_prediction = self.predicted_class is not None

        self.btn_open_dir.setEnabled(has_model)
        self.btn_output_dir.setEnabled(has_model)

        self.btn_next.setEnabled(has_image_dir and self.current_image_index < len(self.image_files) - 1)
        self.btn_accept.setEnabled(has_current_image and has_output_dir and has_prediction)
        self.btn_delete.setEnabled(has_current_image)
        self.btn_manual_move.setEnabled(has_current_image and has_output_dir)
        self.combo_manual_class.setEnabled(has_model and has_current_image)
        
        self.threshold_spinbox.setEnabled(has_model)
        self.auto_delete_switch.setEnabled(has_model)


    def next_image(self):
        # Reset prediction state for the new image
        self.predicted_class = None
        self.lbl_prediction.setText("预测分类: ...")
        
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_files):
            self.show_info("完成", "已处理完所有图片。")
            self.reset_ui_after_completion()
            return

        image_name = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.image_dir, image_name)
        self.lbl_image_path.setText(self.current_image_path)

        try:
            pixmap = QPixmap(self.current_image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            
            results = self.model.predict(self.current_image_path, verbose=False)
            result = results[0]
            probs = result.probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            self.predicted_class = self.class_names[top1_idx]
            self.lbl_prediction.setText(f"预测分类: {self.predicted_class} ({top1_conf:.2f})")

            if self.auto_delete_switch.isChecked() and top1_conf >= self.threshold_spinbox.value():
                self.show_info("自动删除", f"图片 {os.path.basename(self.current_image_path)} 置信度 {top1_conf:.2f}，将在0.5秒后删除。")
                QTimer.singleShot(500, self._perform_auto_delete)
                return

        except Exception as e:
            self.show_info("错误", f"加载或预测图片失败: {e}", success=False)
            self.predicted_class = None
            self.lbl_prediction.setText("预测分类: 错误")

        self._update_button_states()

    def _perform_auto_delete(self):
        if not self.current_image_path:
            return
        try:
            os.remove(self.current_image_path)
            self.image_files.pop(self.current_image_index)
            self.current_image_index -= 1
            self.next_image()
        except Exception as e:
            self.show_info("错误", f"自动删除文件失败: {e}", success=False)
            # Try to advance anyway
            self.image_files.pop(self.current_image_index)
            self.current_image_index -= 1
            self.next_image()

    def accept_prediction(self):
        if not self.predicted_class:
            self.show_info("提示", "没有可接受的预测结果。", success=False)
            return
        self.move_file(self.predicted_class)

    def delete_image(self):
        if not self.current_image_path:
            return
        try:
            deleted_image_name = os.path.basename(self.current_image_path)
            os.remove(self.current_image_path)
            self.show_info("已删除", deleted_image_name)
            self.image_files.pop(self.current_image_index)
            self.current_image_index -= 1
            self.next_image()
        except Exception as e:
            self.show_info("错误", f"删除文件失败: {e}", success=False)

    def manual_move(self):
        selected_class = self.combo_manual_class.currentText()
        if not selected_class or selected_class == "手动选择分类":
            self.show_info("提示", "请先选择一个分类。", success=False)
            return
        self.move_file(selected_class)

    def move_file(self, class_name):
        if not self.current_image_path or not self.output_dir:
            return

        dest_dir = os.path.join(self.output_dir, class_name)
        dest_path = os.path.join(dest_dir, os.path.basename(self.current_image_path))

        try:
            shutil.move(self.current_image_path, dest_path)
            self.show_info("已移动", f"{os.path.basename(self.current_image_path)} -> {class_name}")
            self.image_files.pop(self.current_image_index)
            self.current_image_index -= 1
            self.next_image()
        except Exception as e:
            self.show_info("错误", f"移动文件失败: {e}", success=False)

    def reset_ui_after_completion(self):
        self.image_label.clear()
        self.image_label.setText("完成！请选择新的图片文件夹")
        self.lbl_image_path.setText("图片路径将显示在这里")
        self.lbl_prediction.setText("预测分类: N/A")
        self.current_image_path = None
        self.predicted_class = None
        self.image_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self._update_button_states()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    window = ImageClassifierWindow()
    window.show()
    sys.exit(app.exec())