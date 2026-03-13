"""
ONNX Converter - Standalone app to convert PyTorch models to ONNX
Simple UI to select .pt files and convert them
"""
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QListWidget, 
                             QListWidgetItem, QFileDialog, QTextEdit, QProgressBar,
                             QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor


class ConvertThread(QThread):
    """Thread to convert models without blocking UI"""
    progress = pyqtSignal(str)  # Log message
    finished = pyqtSignal(bool, str)  # Success, message
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
    
    def run(self):
        import io
        import contextlib
        
        try:
            from convert_models_to_onnx import convert_model_to_onnx
            
            self.progress.emit(f"🔄 Converting {Path(self.model_path).name}...")
            
            # Capture stdout to show conversion progress
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                success = convert_model_to_onnx(self.model_path)
            
            # Get captured output
            output = output_buffer.getvalue()
            if output:
                for line in output.split('\n'):
                    if line.strip():
                        self.progress.emit(f"   {line}")
            
            if success:
                onnx_path = self.model_path.replace('.pt', '.onnx')
                self.finished.emit(True, f"✅ Created: {Path(onnx_path).name}")
            else:
                self.finished.emit(False, "❌ Conversion failed")
        
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {e}\n{traceback.format_exc()}"
            self.progress.emit(error_msg)
            self.finished.emit(False, f"❌ Error: {e}")


class ONNXConverterApp(QMainWindow):
    """Simple app to convert PyTorch models to ONNX"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ONNX Model Converter")
        self.setMinimumSize(800, 600)
        
        self.selected_files = []
        self.convert_thread = None
        
        self.setup_ui()
        self.check_onnx_status()
    
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🚀 ONNX Model Converter")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #3b82f6; padding: 10px;")
        layout.addWidget(title)
        
        # Info
        info = QLabel("Convert PyTorch (.pt) models to ONNX format for 2-3x speedup")
        info.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(info)
        
        # ONNX Status
        status_group = QGroupBox("ONNX Runtime Status")
        status_layout = QVBoxLayout()
        self.onnx_status_label = QLabel("Checking...")
        self.onnx_status_label.setStyleSheet("padding: 10px; font-size: 12px;")
        status_layout.addWidget(self.onnx_status_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # File selection
        file_group = QGroupBox("Select Models to Convert")
        file_layout = QVBoxLayout()
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_add_file = QPushButton("➕ Add .pt File")
        self.btn_add_file.clicked.connect(self.add_file)
        btn_layout.addWidget(self.btn_add_file)
        
        self.btn_add_folder = QPushButton("📁 Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_clear = QPushButton("🗑️ Clear List")
        self.btn_clear.clicked.connect(self.clear_list)
        btn_layout.addWidget(self.btn_clear)
        
        file_layout.addLayout(btn_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        file_layout.addWidget(self.file_list)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Progress
        progress_group = QGroupBox("Conversion Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #4ade80; font-family: monospace;")
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Convert button
        self.btn_convert = QPushButton("🚀 Convert All to ONNX")
        self.btn_convert.setMinimumHeight(50)
        self.btn_convert.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #94a3b8;
            }
        """)
        self.btn_convert.clicked.connect(self.start_conversion)
        layout.addWidget(self.btn_convert)
    
    def check_onnx_status(self):
        """Check if ONNX Runtime is installed"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            status_text = f"✅ ONNX Runtime installed\n"
            status_text += f"   Providers: {', '.join(providers)}\n"
            
            if 'CUDAExecutionProvider' in providers:
                status_text += "   🚀 GPU Support: YES"
                self.onnx_status_label.setStyleSheet("padding: 10px; background-color: #dcfce7; color: #166534;")
            else:
                status_text += "   ⚠️  GPU Support: NO (CPU only)"
                self.onnx_status_label.setStyleSheet("padding: 10px; background-color: #fef3c7; color: #92400e;")
            
            self.onnx_status_label.setText(status_text)
            self.btn_convert.setEnabled(True)
            
        except ImportError:
            status_text = "❌ ONNX Runtime NOT installed\n"
            status_text += "   Run: FIX_ONNX_GPU.bat to install"
            self.onnx_status_label.setText(status_text)
            self.onnx_status_label.setStyleSheet("padding: 10px; background-color: #fee2e2; color: #991b1b;")
            self.btn_convert.setEnabled(False)
    
    def add_file(self):
        """Add a single .pt file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PyTorch Model (.pt)",
            ".",
            "PyTorch Models (*.pt);;All Files (*.*)"
        )
        
        if file_path and file_path not in self.selected_files:
            self.selected_files.append(file_path)
            self.update_file_list()
    
    def add_folder(self):
        """Add all .pt files from a folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if folder_path:
            for file in Path(folder_path).rglob("*.pt"):
                file_path = str(file)
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
            
            self.update_file_list()
    
    def clear_list(self):
        """Clear the file list"""
        self.selected_files.clear()
        self.file_list.clear()
    
    def update_file_list(self):
        """Update the file list display"""
        self.file_list.clear()
        
        for file_path in self.selected_files:
            onnx_path = file_path.replace('.pt', '.onnx')
            
            if os.path.exists(onnx_path):
                status = "✅ Already converted"
                color = QColor(34, 197, 94)  # Green
            else:
                status = "⏳ Ready to convert"
                color = QColor(251, 191, 36)  # Yellow
            
            item = QListWidgetItem(f"{Path(file_path).name} - {status}")
            item.setForeground(color)
            self.file_list.addItem(item)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        QApplication.processEvents()
    
    def start_conversion(self):
        """Start converting all selected files"""
        if not self.selected_files:
            self.log("⚠️  No files selected!")
            return
        
        self.btn_convert.setEnabled(False)
        self.btn_add_file.setEnabled(False)
        self.btn_add_folder.setEnabled(False)
        self.btn_clear.setEnabled(False)
        
        self.log_text.clear()
        self.log("=" * 60)
        self.log("Starting conversion...")
        self.log("=" * 60)
        
        self.progress_bar.setMaximum(len(self.selected_files))
        self.progress_bar.setValue(0)
        
        self.current_index = 0
        self.success_count = 0
        self.skip_count = 0
        
        self.convert_next()
    
    def convert_next(self):
        """Convert the next file in the list"""
        if self.current_index >= len(self.selected_files):
            self.conversion_complete()
            return
        
        file_path = self.selected_files[self.current_index]
        
        # Check if already converted
        onnx_path = file_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_path):
            self.log(f"\n✅ {Path(file_path).name} already converted")
            self.skip_count += 1
            self.current_index += 1
            self.progress_bar.setValue(self.current_index)
            self.convert_next()
            return
        
        # Convert
        self.log(f"\n🔄 Converting {Path(file_path).name}...")
        
        self.convert_thread = ConvertThread(file_path)
        self.convert_thread.progress.connect(self.log)
        self.convert_thread.finished.connect(self.on_convert_finished)
        self.convert_thread.start()
    
    def on_convert_finished(self, success, message):
        """Handle conversion completion"""
        self.log(f"   {message}")
        
        if success:
            self.success_count += 1
        
        self.current_index += 1
        self.progress_bar.setValue(self.current_index)
        
        # Update file list
        self.update_file_list()
        
        # Convert next
        self.convert_next()
    
    def conversion_complete(self):
        """All conversions complete"""
        self.log("\n" + "=" * 60)
        self.log("CONVERSION COMPLETE")
        self.log("=" * 60)
        self.log(f"Total files: {len(self.selected_files)}")
        self.log(f"Converted: {self.success_count}")
        self.log(f"Skipped: {self.skip_count}")
        
        if self.success_count > 0:
            self.log("\n🚀 Success! Models are ready to use.")
            self.log("   Restart your main app to use ONNX models.")
        
        self.btn_convert.setEnabled(True)
        self.btn_add_file.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self.btn_clear.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = ONNXConverterApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
