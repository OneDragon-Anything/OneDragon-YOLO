import os
from typing import Dict

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, 
                               QProgressBar, QTextEdit, QGroupBox, QGridLayout)
from qfluentwidgets import (PushButton, PrimaryPushButton, BodyLabel,
                            DoubleSpinBox, SubtitleLabel, LineEdit, ComboBox,
                            InfoBar, InfoBarPosition)

from one_dragon_yolo.gui.similarity_processor import ImageSimilarityProcessor


class ImageSimilarityWorker(QThread):
    """图片相似度处理工作线程"""
    
    progress_updated = Signal(int, int, str)  # current, total, message
    log_message = Signal(str)
    finished = Signal(dict)  # 返回处理结果统计
    
    def __init__(self, root_folder: str, similarity_threshold: float, mode: str):
        """
        初始化工作线程

        Args:
            root_folder: 根文件夹路径
            similarity_threshold: 相似度阈值 (0-1)
            mode: 处理模式 ('cross_folder' 或 'within_folder')
        """
        super().__init__()
        self.root_folder = root_folder
        self.similarity_threshold = similarity_threshold
        self.mode = mode
        self.is_cancelled = False
        self.processor = ImageSimilarityProcessor(similarity_threshold)
        
    def cancel(self):
        """取消处理"""
        self.is_cancelled = True
        
    def run(self):
        """执行图片相似度处理"""
        try:
            def progress_callback(current, total, message):
                if not self.is_cancelled:
                    self.progress_updated.emit(current, total, message)

            def log_callback(message):
                if not self.is_cancelled:
                    self.log_message.emit(message)

            if self.mode == 'cross_folder':
                result = self.processor.process_cross_folder_similarity(
                    self.root_folder, progress_callback, log_callback)
            else:
                result = self.processor.process_within_folder_similarity(
                    self.root_folder, progress_callback, log_callback)

            if not self.is_cancelled:
                self.finished.emit(result)
        except Exception as e:
            self.log_message.emit(f"处理过程中发生错误: {str(e)}")




class ImageSimilarityTab(QWidget):
    """图片相似度删除Tab"""
    
    def __init__(self):
        """初始化图片相似度删除Tab"""
        super().__init__()
        self._init_ui()
        self._connect_signals()
        self.worker = None
        
    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title = SubtitleLabel("图片相似度删除工具")
        layout.addWidget(title)
        
        # 文件夹选择
        folder_group = QGroupBox("文件夹选择")
        folder_layout = QVBoxLayout(folder_group)
        
        folder_select_layout = QHBoxLayout()
        self.btn_select_folder = PushButton("选择根文件夹")
        self.folder_path_edit = LineEdit()
        self.folder_path_edit.setPlaceholderText("请选择包含子文件夹的根目录")
        self.folder_path_edit.setReadOnly(True)
        folder_select_layout.addWidget(self.btn_select_folder)
        folder_select_layout.addWidget(self.folder_path_edit, 1)
        folder_layout.addLayout(folder_select_layout)
        
        layout.addWidget(folder_group)
        
        # 设置参数
        settings_group = QGroupBox("处理设置")
        settings_layout = QGridLayout(settings_group)
        
        # 相似度阈值
        settings_layout.addWidget(BodyLabel("相似度阈值:"), 0, 0)
        self.similarity_spinbox = DoubleSpinBox()
        self.similarity_spinbox.setRange(0.1, 1.0)
        self.similarity_spinbox.setSingleStep(0.05)
        self.similarity_spinbox.setValue(0.85)
        self.similarity_spinbox.setDecimals(2)
        self.similarity_spinbox.setSuffix(" (越高越严格)")
        settings_layout.addWidget(self.similarity_spinbox, 0, 1)
        
        # 处理模式
        settings_layout.addWidget(BodyLabel("处理模式:"), 1, 0)
        self.mode_combo = ComboBox()
        self.mode_combo.addItems([
            "跨文件夹比较 (平衡各文件夹图片数量)",
            "文件夹内比较 (删除每个文件夹内的重复图片)"
        ])
        settings_layout.addWidget(self.mode_combo, 1, 1)
        
        layout.addWidget(settings_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.btn_start = PrimaryPushButton("开始处理")
        self.btn_cancel = PushButton("取消处理")
        self.btn_cancel.setEnabled(False)
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_cancel)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志显示
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)
        
    def _connect_signals(self):
        """连接信号和槽"""
        self.btn_select_folder.clicked.connect(self._select_folder)
        self.btn_start.clicked.connect(self._start_processing)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        
    def _select_folder(self):
        """选择根文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择包含子文件夹的根目录"
        )
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            
    def _start_processing(self):
        """开始处理"""
        folder_path = self.folder_path_edit.text().strip()
        if not folder_path or not os.path.exists(folder_path):
            self._show_info("错误", "请先选择有效的根文件夹", success=False)
            return
            
        # 检查是否有子文件夹
        subfolders = [f for f in os.listdir(folder_path) 
                     if os.path.isdir(os.path.join(folder_path, f))]
        if not subfolders:
            self._show_info("错误", "选择的文件夹中没有子文件夹", success=False)
            return
            
        similarity_threshold = self.similarity_spinbox.value()
        mode = 'cross_folder' if self.mode_combo.currentIndex() == 0 else 'within_folder'
        
        # 启动工作线程
        self.worker = ImageSimilarityWorker(folder_path, similarity_threshold, mode)
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.log_message.connect(self._add_log)
        self.worker.finished.connect(self._processing_finished)
        
        # 更新UI状态
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        
        self._add_log("开始处理...")
        self.worker.start()
        
    def _cancel_processing(self):
        """取消处理"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._add_log("正在取消处理...")
            
    def _update_progress(self, current: int, total: int, message: str):
        """更新进度"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.progress_bar.setFormat(f"{message} ({current}/{total})")
            
    def _add_log(self, message: str):
        """添加日志消息"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
        
    def _processing_finished(self, result: Dict):
        """处理完成"""
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # 显示结果统计
        deleted_files = result.get('deleted_files', 0)
        processed_folders = result.get('processed_folders', 0)
        total_comparisons = result.get('total_comparisons', 0)
        
        summary = (f"处理完成！\n"
                  f"删除文件数: {deleted_files}\n"
                  f"处理文件夹数: {processed_folders}\n"
                  f"总比较次数: {total_comparisons}")
        
        self._add_log(summary)
        self._show_info("完成", summary, success=True)
        
    def _show_info(self, title: str, content: str, success: bool = True):
        """显示信息提示"""
        if success:
            InfoBar.success(title, content, duration=3000, 
                          position=InfoBarPosition.TOP, parent=self)
        else:
            InfoBar.error(title, content, duration=5000, 
                        position=InfoBarPosition.TOP, parent=self)
