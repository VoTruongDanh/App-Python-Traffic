"""
ONNX Converter - Standalone app to convert PyTorch models to ONNX.
Supports using an external Python interpreter (outside current venv).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


def _probe_python(python_exe: str):
    """Probe a Python interpreter for required packages."""
    probe_code = """
import json
import sys
import importlib.util

info = {
    'python': sys.executable,
    'torch': importlib.util.find_spec('torch') is not None,
    'ultralytics': importlib.util.find_spec('ultralytics') is not None,
    'onnxruntime': importlib.util.find_spec('onnxruntime') is not None,
    'providers': [],
}

if info['onnxruntime']:
    import onnxruntime as ort
    info['providers'] = ort.get_available_providers()

print(json.dumps(info))
"""
    try:
        result = subprocess.run(
            [python_exe, "-c", probe_code],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return None
        output = (result.stdout or "").strip().splitlines()
        if not output:
            return None
        return json.loads(output[-1])
    except Exception:
        return None


def _discover_python_candidates():
    """Discover candidate Python interpreters on Windows."""
    candidates = []

    def add_candidate(path):
        if not path:
            return
        norm = str(Path(path))
        if norm not in candidates:
            candidates.append(norm)

    add_candidate(sys.executable)
    add_candidate(os.environ.get("CONDA_PYTHON_EXE"))

    which_python = shutil.which("python")
    add_candidate(which_python)

    # Probe py launcher entries when available
    try:
        py_list = subprocess.run(
            ["py", "-0p"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if py_list.returncode == 0:
            for line in (py_list.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-") and " " in line:
                    add_candidate(line.split()[-1])
    except Exception:
        pass

    return candidates


class ConvertThread(QThread):
    """Thread to convert models without blocking UI."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, model_path: str, python_exe: str, script_dir: Path):
        super().__init__()
        self.model_path = model_path
        self.python_exe = python_exe
        self.script_dir = script_dir

    def run(self):
        try:
            model_name = Path(self.model_path).name
            self.progress.emit(f"Converting {model_name} ...")
            self.progress.emit(f"Interpreter: {self.python_exe}")

            convert_code = (
                "import sys; "
                f"sys.path.insert(0, r'{str(self.script_dir)}'); "
                "from convert_models_to_onnx import convert_model_to_onnx; "
                "ok = convert_model_to_onnx(sys.argv[1]); "
                "print('__CONVERT_STATUS__=' + ('1' if ok else '0'))"
            )

            result = subprocess.run(
                [self.python_exe, "-c", convert_code, self.model_path],
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            for line in stdout.splitlines():
                if line.strip():
                    self.progress.emit(f"  {line}")
            for line in stderr.splitlines():
                if line.strip():
                    self.progress.emit(f"  {line}")

            marker = "__CONVERT_STATUS__=1"
            success = marker in stdout

            # Fallback validation by file existence
            if not success:
                onnx_path = self.model_path.replace(".pt", ".onnx")
                success = Path(onnx_path).exists()

            if success:
                onnx_name = Path(self.model_path.replace(".pt", ".onnx")).name
                self.finished.emit(True, f"Created: {onnx_name}")
            else:
                self.finished.emit(False, "Conversion failed")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(False, f"Error: {e}")


class ONNXConverterApp(QMainWindow):
    """Simple app to convert PyTorch models to ONNX."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ONNX Model Converter")
        self.setMinimumSize(820, 620)

        self.selected_files = []
        self.convert_thread = None
        self.python_infos = []
        self.conversion_python = None

        self.setup_ui()
        self.check_onnx_status()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title = QLabel("ONNX Model Converter")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #2563eb; padding: 10px;")
        layout.addWidget(title)

        info = QLabel("Convert PyTorch (.pt) models to ONNX format for faster inference.")
        info.setStyleSheet("color: #4b5563; padding: 5px;")
        layout.addWidget(info)

        status_group = QGroupBox("Runtime Status")
        status_layout = QVBoxLayout()
        self.onnx_status_label = QLabel("Checking...")
        self.onnx_status_label.setStyleSheet("padding: 10px; font-size: 12px;")
        status_layout.addWidget(self.onnx_status_label)

        self.env_label = QLabel("")
        self.env_label.setStyleSheet("padding: 4px 10px; color: #334155; font-size: 11px;")
        self.env_label.setWordWrap(True)
        status_layout.addWidget(self.env_label)

        refresh_btn = QPushButton("Refresh Runtime Scan")
        refresh_btn.clicked.connect(self.check_onnx_status)
        status_layout.addWidget(refresh_btn)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        file_group = QGroupBox("Select Models to Convert")
        file_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.btn_add_file = QPushButton("Add .pt File")
        self.btn_add_file.clicked.connect(self.add_file)
        btn_layout.addWidget(self.btn_add_file)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(self.btn_add_folder)

        self.btn_clear = QPushButton("Clear List")
        self.btn_clear.clicked.connect(self.clear_list)
        btn_layout.addWidget(self.btn_clear)
        file_layout.addLayout(btn_layout)

        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        file_layout.addWidget(self.file_list)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        progress_group = QGroupBox("Conversion Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet(
            "background-color: #111827; color: #86efac; font-family: Consolas, monospace;"
        )
        progress_layout.addWidget(self.log_text)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        self.btn_convert = QPushButton("Convert All to ONNX")
        self.btn_convert.setMinimumHeight(50)
        self.btn_convert.setStyleSheet(
            """
            QPushButton {
                background-color: #2563eb;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #94a3b8;
            }
            """
        )
        self.btn_convert.clicked.connect(self.start_conversion)
        layout.addWidget(self.btn_convert)

    def _scan_python_environments(self):
        infos = []
        for candidate in _discover_python_candidates():
            info = _probe_python(candidate)
            if info:
                info["candidate"] = candidate
                infos.append(info)
        self.python_infos = infos

        # Prefer current interpreter if it can convert, otherwise choose first valid one.
        current = str(Path(sys.executable))
        selected = None
        for item in infos:
            if item.get("python") == current and item.get("torch") and item.get("ultralytics"):
                selected = item
                break
        if selected is None:
            for item in infos:
                if item.get("torch") and item.get("ultralytics"):
                    selected = item
                    break

        self.conversion_python = selected.get("python") if selected else None

    def check_onnx_status(self):
        """Check runtime availability across interpreters."""
        self._scan_python_environments()
        current_python = str(Path(sys.executable))

        current_info = None
        external_onnx = None
        for info in self.python_infos:
            if info.get("python") == current_python:
                current_info = info
            elif info.get("onnxruntime") and external_onnx is None:
                external_onnx = info

        if current_info and current_info.get("onnxruntime"):
            providers = current_info.get("providers", [])
            text = "ONNX Runtime found in current interpreter.\n"
            text += f"Providers: {', '.join(providers) if providers else 'Unknown'}"
            if "CUDAExecutionProvider" in providers:
                self.onnx_status_label.setStyleSheet(
                    "padding: 10px; background-color: #dcfce7; color: #166534;"
                )
            else:
                self.onnx_status_label.setStyleSheet(
                    "padding: 10px; background-color: #fef3c7; color: #92400e;"
                )
            self.onnx_status_label.setText(text)
        elif external_onnx:
            providers = external_onnx.get("providers", [])
            text = "ONNX Runtime not in current interpreter, but found in external environment.\n"
            text += f"External Providers: {', '.join(providers) if providers else 'Unknown'}"
            self.onnx_status_label.setStyleSheet(
                "padding: 10px; background-color: #fef3c7; color: #92400e;"
            )
            self.onnx_status_label.setText(text)
        else:
            text = "ONNX Runtime not detected in scanned interpreters."
            self.onnx_status_label.setStyleSheet(
                "padding: 10px; background-color: #fee2e2; color: #991b1b;"
            )
            self.onnx_status_label.setText(text)

        if self.conversion_python:
            self.env_label.setText(
                f"Conversion Interpreter: {self.conversion_python}\n"
                "App can run conversion without installing packages in this venv."
            )
            self.btn_convert.setEnabled(True)
        else:
            self.env_label.setText(
                "No interpreter found with both 'torch' and 'ultralytics'. "
                "Install them in one environment, then click Refresh Runtime Scan."
            )
            self.btn_convert.setEnabled(False)

    def add_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PyTorch Model (.pt)",
            str(PROJECT_DIR),
            "PyTorch Models (*.pt);;All Files (*.*)",
        )
        if file_path and file_path not in self.selected_files:
            self.selected_files.append(file_path)
            self.update_file_list()

    def add_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", str(PROJECT_DIR))
        if folder_path:
            for file in Path(folder_path).rglob("*.pt"):
                file_path = str(file)
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
            self.update_file_list()

    def clear_list(self):
        self.selected_files.clear()
        self.file_list.clear()

    def update_file_list(self):
        self.file_list.clear()
        for file_path in self.selected_files:
            onnx_path = file_path.replace(".pt", ".onnx")
            if os.path.exists(onnx_path):
                status = "Already converted"
                color = QColor(34, 197, 94)
            else:
                status = "Ready to convert"
                color = QColor(251, 191, 36)
            item = QListWidgetItem(f"{Path(file_path).name} - {status}")
            item.setForeground(color)
            self.file_list.addItem(item)

    def log(self, message):
        self.log_text.append(message)
        QApplication.processEvents()

    def start_conversion(self):
        if not self.selected_files:
            self.log("No files selected.")
            return

        if not self.conversion_python:
            self.log("No conversion interpreter available.")
            self.check_onnx_status()
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
        if self.current_index >= len(self.selected_files):
            self.conversion_complete()
            return

        file_path = self.selected_files[self.current_index]
        onnx_path = file_path.replace(".pt", ".onnx")
        if os.path.exists(onnx_path):
            self.log(f"{Path(file_path).name} already converted")
            self.skip_count += 1
            self.current_index += 1
            self.progress_bar.setValue(self.current_index)
            self.convert_next()
            return

        self.convert_thread = ConvertThread(file_path, self.conversion_python, SCRIPT_DIR)
        self.convert_thread.progress.connect(self.log)
        self.convert_thread.finished.connect(self.on_convert_finished)
        self.convert_thread.start()

    def on_convert_finished(self, success, message):
        self.log(message)
        if success:
            self.success_count += 1

        self.current_index += 1
        self.progress_bar.setValue(self.current_index)
        self.update_file_list()
        self.convert_next()

    def conversion_complete(self):
        self.log("=" * 60)
        self.log("CONVERSION COMPLETE")
        self.log("=" * 60)
        self.log(f"Total files: {len(self.selected_files)}")
        self.log(f"Converted: {self.success_count}")
        self.log(f"Skipped: {self.skip_count}")
        self.log(f"Interpreter: {self.conversion_python}")

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
