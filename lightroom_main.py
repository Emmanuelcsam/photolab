#!/usr/bin/env python3
import sys
import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# Fix Qt platform plugin issue
# Let Qt auto-detect the best platform instead of forcing xcb
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

# Remove conflicting Qt plugin paths
for key in ['QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH']:
    if key in os.environ:
        del os.environ[key]

# Only force xcb if we're on Wayland (where auto-detection might fail)
if os.environ.get('XDG_SESSION_TYPE') == 'wayland' and 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Import PyQt5 first
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Then import cv2 with headless backend to avoid Qt conflicts
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
import cv2
from collections import OrderedDict
import traceback

from image_processor import ImageProcessor
from pytorch_learner import EditLearner

class ImageThumbnail(QLabel):
    clicked = pyqtSignal(str)
    
    def __init__(self, image_path, size=150):
        super().__init__()
        self.image_path = image_path
        self.setFixedSize(size, size)
        self.setScaledContents(True)
        self.setFrameStyle(QFrame.Box)
        self.setCursor(Qt.PointingHandCursor)
        self.load_thumbnail()
        
    def load_thumbnail(self):
        print(f"[LOG] Loading thumbnail: {self.image_path}")
        img = cv2.imread(self.image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            size = self.width()
            if h > w:
                new_h = size
                new_w = int(w * (size / h))
            else:
                new_w = size
                new_h = int(h * (size / w))
            img = cv2.resize(img, (new_w, new_h))
            qimg = QImage(img.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
            print(f"[LOG] Thumbnail clicked: {self.image_path}")

class FunctionWidget(QWidget):
    parameterChanged = pyqtSignal(str, dict)
    addToEditLine = pyqtSignal(str, dict)
    
    def __init__(self, func_name, func_params):
        super().__init__()
        self.func_name = func_name
        self.func_params = func_params
        self.param_widgets = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        header = QHBoxLayout()
        self.label = QLabel(f"<b>{self.func_name}</b>")
        self.label.setStyleSheet("QLabel { background-color: #3498db; color: white; padding: 5px; }")
        header.addWidget(self.label)
        
        self.add_btn = QPushButton("Add to Edit Line")
        self.add_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; }")
        self.add_btn.clicked.connect(self.add_to_edit_line)
        header.addWidget(self.add_btn)
        
        layout.addLayout(header)
        
        params_widget = QWidget()
        params_layout = QGridLayout(params_widget)
        
        row = 0
        for param_name, param_info in self.func_params.items():
            param_label = QLabel(param_name)
            params_layout.addWidget(param_label, row, 0)
            
            if param_info['type'] == 'slider':
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(param_info['min'])
                slider.setMaximum(param_info['max'])
                slider.setValue(param_info['default'])
                slider.valueChanged.connect(lambda v, p=param_name: self.on_param_changed(p, v))
                
                value_label = QLabel(str(param_info['default']))
                slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
                
                params_layout.addWidget(slider, row, 1)
                params_layout.addWidget(value_label, row, 2)
                self.param_widgets[param_name] = slider
                
            elif param_info['type'] == 'combo':
                combo = QComboBox()
                combo.addItems([str(v) for v in param_info['values']])
                combo.setCurrentText(str(param_info['default']))
                combo.currentTextChanged.connect(lambda v, p=param_name: self.on_param_changed(p, v))
                params_layout.addWidget(combo, row, 1, 1, 2)
                self.param_widgets[param_name] = combo
                
            elif param_info['type'] == 'checkbox':
                checkbox = QCheckBox()
                checkbox.setChecked(param_info['default'])
                checkbox.toggled.connect(lambda v, p=param_name: self.on_param_changed(p, v))
                params_layout.addWidget(checkbox, row, 1, 1, 2)
                self.param_widgets[param_name] = checkbox
                
            row += 1
            
        layout.addWidget(params_widget)
        self.setLayout(layout)
        
    def on_param_changed(self, param_name, value):
        print(f"[LOG] Parameter changed: {self.func_name}.{param_name} = {value}")
        self.parameterChanged.emit(self.func_name, self.get_current_params())
        
    def get_current_params(self):
        params = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSlider):
                params[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[param_name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
        return params
        
    def add_to_edit_line(self):
        print(f"[LOG] Adding to edit line: {self.func_name}")
        self.addToEditLine.emit(self.func_name, self.get_current_params())

class EditLineItem(QWidget):
    removed = pyqtSignal(int)
    moved = pyqtSignal(int, int)
    
    def __init__(self, index, func_name, params):
        super().__init__()
        self.index = index
        self.func_name = func_name
        self.params = params
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.label = QLabel(f"{self.index+1}. {self.func_name}")
        self.label.setStyleSheet("QLabel { background-color: #34495e; color: white; padding: 5px; }")
        layout.addWidget(self.label)
        
        params_text = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        self.params_label = QLabel(params_text)
        self.params_label.setStyleSheet("QLabel { color: #7f8c8d; }")
        layout.addWidget(self.params_label)
        
        layout.addStretch()
        
        self.up_btn = QPushButton("↑")
        self.up_btn.clicked.connect(lambda: self.moved.emit(self.index, self.index - 1))
        layout.addWidget(self.up_btn)
        
        self.down_btn = QPushButton("↓")
        self.down_btn.clicked.connect(lambda: self.moved.emit(self.index, self.index + 1))
        layout.addWidget(self.down_btn)
        
        self.remove_btn = QPushButton("×")
        self.remove_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self.index))
        layout.addWidget(self.remove_btn)
        
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print("[LOG] Initializing MainWindow")
        self.processor = ImageProcessor()
        self.learner = EditLearner()
        self.current_image_path = None
        self.original_image = None
        self.preview_image = None
        self.edit_line = []
        self.undo_stack = []
        self.image_history = {}
        self.setup_ui()
        self.setup_functions()
        
    def setup_ui(self):
        print("[LOG] Setting up UI")
        self.setWindowTitle("OpenCV Lightroom - Advanced Photo Editor")
        self.setGeometry(100, 100, 1600, 900)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QHBoxLayout(self.central_widget)
        
        left_panel = QVBoxLayout()
        
        dir_btn = QPushButton("Select Directory")
        dir_btn.clicked.connect(self.select_directory)
        left_panel.addWidget(dir_btn)
        
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.IconMode)
        self.image_list.setIconSize(QSize(150, 150))
        self.image_list.setMovement(QListWidget.Static)
        self.image_list.setSpacing(10)
        left_panel.addWidget(QLabel("Images:"))
        left_panel.addWidget(self.image_list)
        
        main_layout.addLayout(left_panel, 1)
        
        center_panel = QVBoxLayout()
        
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("<b>Preview:</b>"))
        
        self.auto_mode_btn = QPushButton("Automatic Mode")
        self.auto_mode_btn.setStyleSheet("QPushButton { background-color: #9b59b6; color: white; }")
        self.auto_mode_btn.clicked.connect(self.apply_automatic_mode)
        preview_header.addWidget(self.auto_mode_btn)
        
        center_panel.addLayout(preview_header)
        
        self.image_viewer = QLabel()
        self.image_viewer.setMinimumSize(800, 600)
        self.image_viewer.setScaledContents(True)
        self.image_viewer.setFrameStyle(QFrame.Box)
        self.image_viewer.setAlignment(Qt.AlignCenter)
        center_panel.addWidget(self.image_viewer)
        
        controls = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-size: 16px; padding: 10px; }")
        self.apply_btn.clicked.connect(self.apply_changes)
        controls.addWidget(self.apply_btn)
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        controls.addWidget(self.undo_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_image)
        controls.addWidget(self.reset_btn)
        
        self.export_btn = QPushButton("Export Image")
        self.export_btn.clicked.connect(self.export_image)
        controls.addWidget(self.export_btn)
        
        self.save_edits_btn = QPushButton("Save Edits")
        self.save_edits_btn.clicked.connect(self.save_edits)
        controls.addWidget(self.save_edits_btn)
        
        self.load_edits_btn = QPushButton("Load Edits")
        self.load_edits_btn.clicked.connect(self.load_edits)
        controls.addWidget(self.load_edits_btn)
        
        center_panel.addLayout(controls)
        
        center_panel.addWidget(QLabel("<b>Edit Line:</b>"))
        self.edit_line_widget = QWidget()
        self.edit_line_layout = QVBoxLayout(self.edit_line_widget)
        
        scroll = QScrollArea()
        scroll.setWidget(self.edit_line_widget)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        center_panel.addWidget(scroll)
        
        main_layout.addLayout(center_panel, 2)
        
        right_panel = QVBoxLayout()
        
        right_panel.addWidget(QLabel("<b>Functions:</b>"))
        
        self.function_tabs = QTabWidget()
        right_panel.addWidget(self.function_tabs)
        
        main_layout.addLayout(right_panel, 1)
        
        self.create_menu_bar()
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Directory', self)
        open_action.triggered.connect(self.select_directory)
        file_menu.addAction(open_action)
        
        export_all_action = QAction('Export All', self)
        export_all_action.triggered.connect(self.export_all_images)
        file_menu.addAction(export_all_action)
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def setup_functions(self):
        print("[LOG] Setting up function widgets")
        
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        basic_scroll = QScrollArea()
        basic_content = QWidget()
        basic_content_layout = QVBoxLayout(basic_content)
        
        for func_name, params in self.processor.get_basic_adjustments().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            basic_content_layout.addWidget(widget)
            
        basic_content_layout.addStretch()
        basic_scroll.setWidget(basic_content)
        basic_scroll.setWidgetResizable(True)
        basic_layout.addWidget(basic_scroll)
        basic_tab.setLayout(basic_layout)
        self.function_tabs.addTab(basic_tab, "Basic")
        
        filters_tab = QWidget()
        filters_layout = QVBoxLayout()
        filters_scroll = QScrollArea()
        filters_content = QWidget()
        filters_content_layout = QVBoxLayout(filters_content)
        
        for func_name, params in self.processor.get_filter_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            filters_content_layout.addWidget(widget)
            
        filters_content_layout.addStretch()
        filters_scroll.setWidget(filters_content)
        filters_scroll.setWidgetResizable(True)
        filters_layout.addWidget(filters_scroll)
        filters_tab.setLayout(filters_layout)
        self.function_tabs.addTab(filters_tab, "Filters")
        
        threshold_tab = QWidget()
        threshold_layout = QVBoxLayout()
        threshold_scroll = QScrollArea()
        threshold_content = QWidget()
        threshold_content_layout = QVBoxLayout(threshold_content)
        
        for func_name, params in self.processor.get_threshold_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            threshold_content_layout.addWidget(widget)
            
        threshold_content_layout.addStretch()
        threshold_scroll.setWidget(threshold_content)
        threshold_scroll.setWidgetResizable(True)
        threshold_layout.addWidget(threshold_scroll)
        threshold_tab.setLayout(threshold_layout)
        self.function_tabs.addTab(threshold_tab, "Threshold")
        
        morph_tab = QWidget()
        morph_layout = QVBoxLayout()
        morph_scroll = QScrollArea()
        morph_content = QWidget()
        morph_content_layout = QVBoxLayout(morph_content)
        
        for func_name, params in self.processor.get_morphology_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            morph_content_layout.addWidget(widget)
            
        morph_content_layout.addStretch()
        morph_scroll.setWidget(morph_content)
        morph_scroll.setWidgetResizable(True)
        morph_layout.addWidget(morph_scroll)
        morph_tab.setLayout(morph_layout)
        self.function_tabs.addTab(morph_tab, "Morphology")
        
        edge_tab = QWidget()
        edge_layout = QVBoxLayout()
        edge_scroll = QScrollArea()
        edge_content = QWidget()
        edge_content_layout = QVBoxLayout(edge_content)
        
        for func_name, params in self.processor.get_edge_detection_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            edge_content_layout.addWidget(widget)
            
        edge_content_layout.addStretch()
        edge_scroll.setWidget(edge_content)
        edge_scroll.setWidgetResizable(True)
        edge_layout.addWidget(edge_scroll)
        edge_tab.setLayout(edge_layout)
        self.function_tabs.addTab(edge_tab, "Edge Detection")
        
        enhance_tab = QWidget()
        enhance_layout = QVBoxLayout()
        enhance_scroll = QScrollArea()
        enhance_content = QWidget()
        enhance_content_layout = QVBoxLayout(enhance_content)
        
        for func_name, params in self.processor.get_enhancement_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            enhance_content_layout.addWidget(widget)
            
        enhance_content_layout.addStretch()
        enhance_scroll.setWidget(enhance_content)
        enhance_scroll.setWidgetResizable(True)
        enhance_layout.addWidget(enhance_scroll)
        enhance_tab.setLayout(enhance_layout)
        self.function_tabs.addTab(enhance_tab, "Enhancement")
        
        color_tab = QWidget()
        color_layout = QVBoxLayout()
        color_scroll = QScrollArea()
        color_content = QWidget()
        color_content_layout = QVBoxLayout(color_content)
        
        for func_name, params in self.processor.get_color_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            color_content_layout.addWidget(widget)
            
        color_content_layout.addStretch()
        color_scroll.setWidget(color_content)
        color_scroll.setWidgetResizable(True)
        color_layout.addWidget(color_scroll)
        color_tab.setLayout(color_layout)
        self.function_tabs.addTab(color_tab, "Color")
        
        artistic_tab = QWidget()
        artistic_layout = QVBoxLayout()
        artistic_scroll = QScrollArea()
        artistic_content = QWidget()
        artistic_content_layout = QVBoxLayout(artistic_content)
        
        for func_name, params in self.processor.get_artistic_functions().items():
            widget = FunctionWidget(func_name, params)
            widget.parameterChanged.connect(self.on_function_parameter_changed)
            widget.addToEditLine.connect(self.add_function_to_edit_line)
            artistic_content_layout.addWidget(widget)
            
        artistic_content_layout.addStretch()
        artistic_scroll.setWidget(artistic_content)
        artistic_scroll.setWidgetResizable(True)
        artistic_layout.addWidget(artistic_scroll)
        artistic_tab.setLayout(artistic_layout)
        self.function_tabs.addTab(artistic_tab, "Artistic")
        
    def select_directory(self):
        print("[LOG] Opening directory selection dialog")
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            print(f"[LOG] Directory selected: {directory}")
            self.load_images_from_directory(directory)
            
    def load_images_from_directory(self, directory):
        print(f"[LOG] Loading images from directory: {directory}")
        self.image_list.clear()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        for file_path in Path(directory).iterdir():
            if file_path.suffix.lower() in image_extensions:
                print(f"[LOG] Found image: {file_path}")
                thumbnail = ImageThumbnail(str(file_path))
                thumbnail.clicked.connect(self.load_image)
                
                item = QListWidgetItem()
                item.setSizeHint(QSize(160, 160))
                self.image_list.addItem(item)
                self.image_list.setItemWidget(item, thumbnail)
                
    def load_image(self, image_path):
        print(f"[LOG] Loading image: {image_path}")
        self.current_image_path = image_path
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            print(f"[LOG] ERROR: Failed to load image: {image_path}")
            return
            
        self.preview_image = self.original_image.copy()
        self.undo_stack = []
        self.edit_line = []
        self.update_edit_line_display()
        self.display_image(self.preview_image)
        self.status_bar.showMessage(f"Loaded: {os.path.basename(image_path)}")
        
    def display_image(self, image):
        if image is None:
            return
            
        print("[LOG] Displaying image")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        max_size = 800
        if h > max_size or w > max_size:
            scale = min(max_size/h, max_size/w)
            new_h, new_w = int(h*scale), int(w*scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h))
            h, w = new_h, new_w
            bytes_per_line = ch * w
            
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_viewer.setPixmap(QPixmap.fromImage(qt_image))
        
    def on_function_parameter_changed(self, func_name, params):
        print(f"[LOG] Function parameter changed: {func_name}")
        self.update_preview()
        
    def add_function_to_edit_line(self, func_name, params):
        print(f"[LOG] Adding function to edit line: {func_name}")
        self.edit_line.append({'function': func_name, 'params': params})
        self.update_edit_line_display()
        self.update_preview()
        
    def update_edit_line_display(self):
        print("[LOG] Updating edit line display")
        for i in reversed(range(self.edit_line_layout.count())):
            self.edit_line_layout.itemAt(i).widget().deleteLater()
            
        for i, edit in enumerate(self.edit_line):
            item = EditLineItem(i, edit['function'], edit['params'])
            item.removed.connect(self.remove_from_edit_line)
            item.moved.connect(self.move_in_edit_line)
            self.edit_line_layout.addWidget(item)
            
    def remove_from_edit_line(self, index):
        print(f"[LOG] Removing from edit line at index: {index}")
        if 0 <= index < len(self.edit_line):
            self.edit_line.pop(index)
            self.update_edit_line_display()
            self.update_preview()
            
    def move_in_edit_line(self, from_index, to_index):
        print(f"[LOG] Moving in edit line from {from_index} to {to_index}")
        if 0 <= from_index < len(self.edit_line) and 0 <= to_index < len(self.edit_line):
            self.edit_line[from_index], self.edit_line[to_index] = self.edit_line[to_index], self.edit_line[from_index]
            self.update_edit_line_display()
            self.update_preview()
            
    def update_preview(self):
        if self.original_image is None:
            return
            
        print("[LOG] Updating preview")
        try:
            self.preview_image = self.processor.apply_edit_line(self.original_image, self.edit_line)
            self.display_image(self.preview_image)
        except Exception as e:
            print(f"[LOG] ERROR in preview update: {str(e)}")
            traceback.print_exc()
            
    def apply_changes(self):
        if self.preview_image is None:
            return
            
        print("[LOG] Applying changes")
        self.undo_stack.append(self.original_image.copy())
        self.original_image = self.preview_image.copy()
        
        if self.current_image_path:
            self.image_history[self.current_image_path] = self.edit_line.copy()
            self.learner.add_edit_sequence(self.current_image_path, self.edit_line)
            
        self.edit_line = []
        self.update_edit_line_display()
        self.status_bar.showMessage("Changes applied")
        
    def undo(self):
        if not self.undo_stack:
            return
            
        print("[LOG] Undoing last action")
        self.original_image = self.undo_stack.pop()
        self.preview_image = self.original_image.copy()
        self.display_image(self.preview_image)
        self.status_bar.showMessage("Undo successful")
        
    def reset_image(self):
        if self.current_image_path:
            print("[LOG] Resetting image")
            self.load_image(self.current_image_path)
            self.status_bar.showMessage("Image reset")
            
    def export_image(self):
        if self.preview_image is None:
            return
            
        print("[LOG] Exporting image")
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            cv2.imwrite(file_path, self.preview_image)
            print(f"[LOG] Image exported to: {file_path}")
            self.status_bar.showMessage(f"Exported: {file_path}")
            
    def export_all_images(self):
        print("[LOG] Exporting all images")
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            export_count = 0
            for i in range(self.image_list.count()):
                widget = self.image_list.itemWidget(self.image_list.item(i))
                if isinstance(widget, ImageThumbnail):
                    image_path = widget.image_path
                    image = cv2.imread(image_path)
                    
                    if image_path in self.image_history:
                        processed = self.processor.apply_edit_line(image, self.image_history[image_path])
                    else:
                        processed = image
                        
                    export_name = f"edited_{os.path.basename(image_path)}"
                    export_path = os.path.join(directory, export_name)
                    cv2.imwrite(export_path, processed)
                    export_count += 1
                    print(f"[LOG] Exported: {export_path}")
                    
            self.status_bar.showMessage(f"Exported {export_count} images")
            
    def save_edits(self):
        print("[LOG] Saving edits")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Edits", "", "Edit Files (*.edit)")
        if file_path:
            edit_data = {
                'edit_line': self.edit_line,
                'history': self.image_history,
                'timestamp': datetime.now().isoformat()
            }
            with open(file_path, 'w') as f:
                json.dump(edit_data, f, indent=2)
            print(f"[LOG] Edits saved to: {file_path}")
            self.status_bar.showMessage(f"Edits saved: {file_path}")
            
    def load_edits(self):
        print("[LOG] Loading edits")
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Edits", "", "Edit Files (*.edit)")
        if file_path:
            with open(file_path, 'r') as f:
                edit_data = json.load(f)
            self.edit_line = edit_data.get('edit_line', [])
            self.image_history = edit_data.get('history', {})
            self.update_edit_line_display()
            self.update_preview()
            print(f"[LOG] Edits loaded from: {file_path}")
            self.status_bar.showMessage(f"Edits loaded: {file_path}")
            
    def apply_automatic_mode(self):
        if not self.current_image_path:
            return
            
        print("[LOG] Applying automatic mode")
        self.status_bar.showMessage("Applying learned edits...")
        
        suggested_edits = self.learner.suggest_edits(self.current_image_path)
        if suggested_edits:
            self.edit_line = suggested_edits
            self.update_edit_line_display()
            self.update_preview()
            self.status_bar.showMessage("Automatic edits applied")
        else:
            self.status_bar.showMessage("No learned edits available")

def main():
    print("[LOG] Starting OpenCV Lightroom application")
    app = QApplication(sys.argv)
    
    # Force X11 backend on Linux
    app.setAttribute(Qt.AA_X11InitThreads, True)
    
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()