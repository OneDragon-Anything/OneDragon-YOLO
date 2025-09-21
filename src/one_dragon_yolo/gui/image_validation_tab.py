import sys
import os
import shutil
import configparser
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog

from qfluentwidgets import (PushButton, PrimaryPushButton, BodyLabel, ComboBox,
                            InfoBar, InfoBarPosition, SubtitleLabel, LineEdit, FluentIcon)
from ultralytics import YOLO


class ValidationWorker(QThread):
    """校验工作线程"""
    # 定义信号
    progress_signal = Signal(int, int)  # 当前进度, 总数
    file_found_signal = Signal(str, str)  # 文件路径, 当前分类
    prediction_result = Signal(str, str, float)  # 预测分类, 当前分类, 置信度
    file_validated = Signal(bool, str)  # 是否需要校验, 文件路径
    error_signal = Signal(str)  # 错误信息
    completed_signal = Signal()  # 完成信号
    request_user_action = Signal(str, str, str, str)  # 文件路径, 当前分类, 预测分类, 置信度

    def __init__(self, model, validation_dir):
        super().__init__()
        self.model = model
        self.validation_dir = validation_dir
        self.class_names = model.names if isinstance(model.names, dict) else {i: name for i, name in enumerate(model.names)}
        self.all_files = []
        self.is_running = True
        self.wait_for_user = False

    def stop(self):
        """停止工作线程"""
        self.is_running = False

    def continue_processing(self):
        """继续处理（用户操作后调用）"""
        self.wait_for_user = False

    def run(self):
        """工作线程主函数"""
        try:
            # 收集文件
            self.all_files = []
            self._collect_image_files(self.validation_dir)

            if not self.all_files:
                self.error_signal.emit("没有找到任何图片文件")
                return

            self.progress_signal.emit(0, len(self.all_files))

            # 处理每个文件
            for i, file_info in enumerate(self.all_files):
                if not self.is_running:
                    break

                file_path = file_info['path']
                current_class = file_info['class']

                self.file_found_signal.emit(file_path, current_class)

                # 进行预测
                try:
                    results = self.model.predict(file_path, verbose=False)
                    result = results[0]
                    probs = result.probs
                    top1_idx = probs.top1
                    top1_conf = probs.top1conf.item()
                    predicted_class = self.class_names[top1_idx]

                    self.prediction_result.emit(predicted_class, current_class, top1_conf)

                    # 检查是否需要校验
                    if predicted_class == current_class:
                        # 分类一致，自动跳过
                        self.file_validated.emit(False, file_path)
                    else:
                        # 分类不一致，等待用户操作
                        self.file_validated.emit(True, file_path)
                        self.request_user_action.emit(file_path, current_class, predicted_class, f"{top1_conf:.2f}")

                        # 等待用户操作
                        self.wait_for_user = True
                        while self.wait_for_user and self.is_running:
                            self.msleep(100)  # 100ms检查一次

                except Exception as e:
                    self.error_signal.emit(f"预测文件 {os.path.basename(file_path)} 失败: {str(e)}")
                    self.file_validated.emit(True, file_path)  # 预测错误也需要用户处理

                self.progress_signal.emit(i + 1, len(self.all_files))

            self.completed_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"校验过程出错: {str(e)}")

    def _collect_image_files(self, directory):
        """递归收集目录下的所有图片文件"""
        for root, dirs, files in os.walk(directory):
            # 跳过所有隐藏文件夹（以.开头的文件夹）
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    # 获取文件所在的文件夹名作为当前分类
                    current_class = os.path.basename(os.path.dirname(file_path))
                    self.all_files.append({
                        'path': file_path,
                        'class': current_class
                    })


class ImageValidationTab(QWidget):
    def __init__(self):
        super().__init__()

        # --- Configuration Management ---
        self.config_dir = os.path.join(os.getcwd(), ".conf")
        self.config_file = os.path.join(self.config_dir, "validation_gui.ini")
        self.config = configparser.ConfigParser()

        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)

        # 加载配置
        self.load_config()

        # --- State Variables ---
        self.validation_dir = ""
        self.model = None
        self.class_names = {}
        self.all_files = []
        self.current_file_index = 0
        self.current_file_path = None
        self.current_file_class = None  # 当前文件所在文件夹的分类
        self.predicted_class = None
        self.is_validating = False
        self.validation_worker = None  # 工作线程

        # --- Main Widget and Layout ---
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

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

        # Validation directory selection
        validation_layout = QHBoxLayout()
        self.btn_select_validation_dir = PushButton("选择待校验目录")
        self.validation_dir_edit = LineEdit()
        self.validation_dir_edit.setPlaceholderText("选择包含子文件夹的输出目录")
        self.validation_dir_edit.setReadOnly(True)
        validation_layout.addWidget(self.btn_select_validation_dir)
        validation_layout.addWidget(self.validation_dir_edit, 1)
        self.main_layout.addLayout(validation_layout)

        # Start validation button
        self.btn_start_validation = PrimaryPushButton("开始校验")
        self.main_layout.addWidget(self.btn_start_validation)

        # Image display
        self.image_label = QLabel("请先加载模型，选择待校验目录，然后开始校验")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #2C2C2C; border-radius: 5px;")
        self.main_layout.addWidget(self.image_label, 1)

        # File info
        info_layout = QVBoxLayout()
        self.lbl_file_path = BodyLabel("文件路径: 未选择")
        self.lbl_file_path.setWordWrap(True)
        self.lbl_current_class = BodyLabel("当前分类: 未确定")
        self.lbl_prediction = BodyLabel("预测分类: 未确定")
        info_layout.addWidget(self.lbl_file_path)
        info_layout.addWidget(self.lbl_current_class)
        info_layout.addWidget(self.lbl_prediction)
        self.main_layout.addLayout(info_layout)

        # Action buttons
        action_layout = QHBoxLayout()
        self.btn_move_to_predicted = PrimaryPushButton("移动到预测分类")
        self.btn_select_and_move = PushButton("选择分类并移动")
        self.btn_skip = PushButton("跳过此文件")
        self.btn_delete = PushButton("删除当前文件")
        action_layout.addWidget(self.btn_move_to_predicted)
        action_layout.addWidget(self.btn_select_and_move)
        action_layout.addWidget(self.btn_skip)
        action_layout.addWidget(self.btn_delete)
        self.main_layout.addLayout(action_layout)

        # Manual classification selection
        manual_layout = QHBoxLayout()
        self.combo_manual_class = ComboBox()
        self.combo_manual_class.setPlaceholderText("选择目标分类")
        self.btn_manual_move = PushButton("移动到所选分类")
        manual_layout.addWidget(self.combo_manual_class)
        manual_layout.addWidget(self.btn_manual_move)
        self.main_layout.addLayout(manual_layout)

        # --- Connections ---
        self.btn_select_model.clicked.connect(self.select_and_load_model)
        self.btn_select_validation_dir.clicked.connect(self.select_validation_dir)
        self.btn_start_validation.clicked.connect(self.start_validation)
        self.btn_move_to_predicted.clicked.connect(self.move_to_predicted_class)
        self.btn_select_and_move.clicked.connect(self.select_class_and_move)
        self.btn_skip.clicked.connect(self.skip_current_file)
        self.btn_delete.clicked.connect(self.delete_current_file)
        self.btn_manual_move.clicked.connect(self.manual_move)

        # --- Thread Signal Connections ---
        # 工作线程信号将在创建时连接

        # --- Initial State ---
        self._update_button_states()

    def show_info(self, title, content, success=True):
        InfoBar.success(title, content, duration=3000, position=InfoBarPosition.TOP, parent=self) if success else InfoBar.error(title, content, duration=5000, position=InfoBarPosition.TOP, parent=self)

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

    def select_validation_dir(self):
        # 从配置中获取上次选择的校验目录作为默认路径
        last_validation_dir = self.get_config_value("last_validation_dir")
        if not os.path.exists(last_validation_dir):
            last_validation_dir = ""

        dir_path = QFileDialog.getExistingDirectory(self, "选择待校验目录", last_validation_dir)
        if not dir_path:
            return

        # 保存选择的校验目录
        self.set_config_value("last_validation_dir", dir_path)
        self.validation_dir = dir_path
        self.validation_dir_edit.setText(self.validation_dir)
        self.show_info("成功", f"待校验目录: {self.validation_dir}")
        self._update_button_states()

    def start_validation(self):
        if not self.model or not self.validation_dir:
            self.show_info("错误", "请先加载模型并选择待校验目录", success=False)
            return

        # 停止现有的工作线程
        if self.validation_worker and self.validation_worker.isRunning():
            self.validation_worker.stop()
            self.validation_worker.wait()

        # 创建新的工作线程
        self.validation_worker = ValidationWorker(self.model, self.validation_dir)

        # 连接信号
        self.validation_worker.progress_signal.connect(self._on_progress)
        self.validation_worker.file_found_signal.connect(self._on_file_found)
        self.validation_worker.prediction_result.connect(self._on_prediction_result)
        self.validation_worker.file_validated.connect(self._on_file_validated)
        self.validation_worker.error_signal.connect(self._on_error)
        self.validation_worker.completed_signal.connect(self._on_completed)
        self.validation_worker.request_user_action.connect(self._on_request_user_action)

        self.is_validating = True
        self.show_info("开始校验", "正在收集文件...")
        self.validation_worker.start()

    # 线程化后不再需要旧的验证循环方法
    # _validation_loop 和 _process_current_file 已被工作线程替代

    # === 工作线程槽函数 ===
    def _on_progress(self, current, total):
        """处理进度信号"""
        pass  # 暂时不显示进度，避免频繁更新UI

    def _on_file_found(self, file_path, current_class):
        """处理文件发现信号"""
        self.current_file_path = file_path
        self.current_file_class = current_class

    def _on_prediction_result(self, predicted_class, current_class, confidence):
        """处理预测结果信号"""
        self.predicted_class = predicted_class

    def _on_file_validated(self, needs_validation, file_path):
        """处理文件校验结果信号"""
        if not needs_validation:
            # 分类一致，自动跳过，继续处理
            if self.validation_worker:
                self.validation_worker.continue_processing()

    def _on_error(self, error_msg):
        """处理错误信号"""
        self.show_info("错误", error_msg, success=False)

    def _on_completed(self):
        """处理完成信号"""
        self.show_info("完成", "校验完成！")
        self.reset_ui_after_completion()

    def _on_request_user_action(self, file_path, current_class, predicted_class, confidence):
        """处理需要用户操作的信号"""
        self.update_file_display()
        self.lbl_prediction.setText(f"预测分类: {predicted_class} ({confidence})")
        self.show_info("需要校验", f"预测分类({predicted_class})与当前分类({current_class})不一致")

    # _collect_files 方法已移到工作线程中
    # _continue_validation 方法已被工作线程机制替代

    def update_file_display(self):
        """更新文件显示"""
        self.lbl_file_path.setText(f"文件路径: {self.current_file_path}")
        self.lbl_current_class.setText(f"当前分类: {self.current_file_class}")
        self.lbl_prediction.setText("预测分类: 正在预测...")

        try:
            pixmap = QPixmap(self.current_file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            self.image_label.clear()
            self.image_label.setText("图片加载失败")

        self._update_button_states()

    # predict_current_file 方法已被 _process_current_file 替代

    def move_to_predicted_class(self):
        """移动到预测的分类"""
        if not self.predicted_class:
            self.show_info("错误", "没有可用的预测分类", success=False)
            return
        self.move_file(self.predicted_class)

    def select_class_and_move(self):
        """选择分类并移动"""
        if not self.model or not self.class_names:
            self.show_info("错误", "请先加载模型", success=False)
            return

        # 让用户选择分类
        selected_class = self.combo_manual_class.currentText()
        if not selected_class or selected_class == "选择目标分类":
            self.show_info("提示", "请先选择一个目标分类", success=False)
            return
        self.move_file(selected_class)

    def manual_move(self):
        """手动移动到选择的分类"""
        selected_class = self.combo_manual_class.currentText()
        if not selected_class or selected_class == "选择目标分类":
            self.show_info("提示", "请先选择一个目标分类", success=False)
            return
        self.move_file(selected_class)

    def move_file(self, target_class):
        """移动文件到目标分类"""
        if not self.current_file_path or not target_class:
            return

        # 确保目标分类目录存在
        target_dir = os.path.join(self.validation_dir, target_class)
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            self.show_info("错误", f"创建目标目录失败: {e}", success=False)
            return

        # 构建目标路径
        target_path = os.path.join(target_dir, os.path.basename(self.current_file_path))

        try:
            shutil.move(self.current_file_path, target_path)
            self.show_info("已移动", f"{os.path.basename(self.current_file_path)} -> {target_class}")
            # 通知工作线程继续处理
            if self.validation_worker:
                self.validation_worker.continue_processing()
        except Exception as e:
            self.show_info("错误", f"移动文件失败: {e}", success=False)

    def skip_current_file(self):
        """跳过当前文件"""
        # 通知工作线程继续处理
        if self.validation_worker:
            self.validation_worker.continue_processing()

    def delete_current_file(self):
        """删除当前文件"""
        if not self.current_file_path:
            return

        try:
            deleted_file_name = os.path.basename(self.current_file_path)
            os.remove(self.current_file_path)
            self.show_info("已删除", deleted_file_name)
            # 通知工作线程继续处理
            if self.validation_worker:
                self.validation_worker.continue_processing()
        except Exception as e:
            self.show_info("错误", f"删除文件失败: {e}", success=False)
            # 即使删除失败也要继续处理
            if self.validation_worker:
                self.validation_worker.continue_processing()

    def reset_ui_after_completion(self):
        """重置UI完成状态"""
        # 停止并清理工作线程
        if self.validation_worker:
            self.validation_worker.stop()
            if self.validation_worker.isRunning():
                self.validation_worker.wait()
            self.validation_worker = None

        self.image_label.clear()
        self.image_label.setText("校验完成！请选择新的待校验目录")
        self.lbl_file_path.setText("文件路径: 未选择")
        self.lbl_current_class.setText("当前分类: 未确定")
        self.lbl_prediction.setText("预测分类: 未确定")
        self.current_file_path = None
        self.current_file_class = None
        self.predicted_class = None
        self.all_files = []
        self.current_file_index = 0
        self.is_validating = False
        self._update_button_states()

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
                'last_validation_dir': ''
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

    def _update_button_states(self):
        """更新按钮状态"""
        has_model = self.model is not None
        has_validation_dir = self.validation_dir != ""
        has_current_file = self.current_file_path is not None and os.path.exists(self.current_file_path)
        has_prediction = self.predicted_class is not None
        is_validating = self.is_validating

        self.btn_select_validation_dir.setEnabled(has_model)
        self.btn_start_validation.setEnabled(has_model and has_validation_dir and not is_validating)

        self.btn_move_to_predicted.setEnabled(has_current_file and has_prediction and is_validating)
        self.btn_select_and_move.setEnabled(has_current_file and has_prediction and is_validating)
        self.btn_skip.setEnabled(has_current_file and is_validating)
        self.btn_delete.setEnabled(has_current_file and is_validating)
        self.btn_manual_move.setEnabled(has_current_file and is_validating)
        self.combo_manual_class.setEnabled(has_model and has_current_file)