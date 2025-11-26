import os
import tempfile

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2

os.environ["QT_XCB_NO_XKB"] = "1"
if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import math
from PyQt5 import QtWidgets, QtCore, QtGui
from camera_utils import get_available_cameras
import numpy as np


class MultiCamRecorder(QtWidgets.QWidget):
    def __init__(self, target_fps=20, target_res=(640, 480), n_required_cameras=None, api_preference=cv2.CAP_V4L2):
        super().__init__()
        self.setWindowTitle("Multi-Camera Recorder")

        self.resize(1280, 720)

        self.target_fps = target_fps
        self.target_res = target_res
        self.frame_delay = int(1000 / target_fps)  # in ms
        self.is_recording = False

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.video_label = QtWidgets.QLabel()
        self.layout.addWidget(self.video_label)

        self.toggle_button = QtWidgets.QPushButton("Start Recording")
        self.toggle_button.clicked.connect(self.toggle_recording)
        self.layout.addWidget(self.toggle_button)

        self.camera_indices = []
        self.camera_ids = []

        for camera_info in get_available_cameras(api_preference):
            self.camera_indices.append(camera_info.index)
            self.camera_ids.append(f"{camera_info.pid}_{camera_info.vid}_{camera_info.index}")

        if n_required_cameras is not None and len(self.camera_indices) < n_required_cameras:
            raise Exception(
                f"Not enough cameras available. {n_required_cameras} required, {len(self.camera_indices)} available."
            )

        self.webcams = [cv2.VideoCapture(c) for c in self.camera_indices]
        if not self.webcams:
            raise Exception("No webcams available")

        self.temp_videos = [
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            for _ in range(len(self.webcams))
        ]
        self.writers = []

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.frame_delay)

    def start_recording(self):
        self.writers = [
            cv2.VideoWriter(
                self.temp_videos[i].name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.target_fps,
                self.target_res
            )
            for i in range(len(self.webcams))
        ]

        self.is_recording = True
        self.toggle_button.setText("Stop Recording")
        print("Recording started")

    def stop_recording(self):
        if self.is_recording:
            for writer in self.writers:
                writer.release()
            for tmp_video in self.temp_videos:
                tmp_video.close()
            self.writers = []
            self.is_recording = False
            print("Recording stopped")

        self.close()

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.toggle_button.setEnabled(False)
            self.stop_recording()

    def update_frame(self):
        frames = []

        for i, webcam in enumerate(self.webcams):

            if not webcam.isOpened():
                continue

            ret, frame = webcam.read()
            if not ret:
                continue
            frame = cv2.resize(frame, self.target_res)

            if self.is_recording and i < len(self.writers):
                self.writers[i].write(frame)

            frames.append(frame)

        if frames:
            n_frames = len(frames)
            grid_cols = math.ceil(math.sqrt(n_frames))
            grid_rows = math.ceil(n_frames / grid_cols)

            blank_frame = np.zeros_like(frames[0])

            mosaic_rows = []
            for r in range(grid_rows):
                row_frames = frames[r * grid_cols:(r + 1) * grid_cols]
                while len(row_frames) < grid_cols:
                    row_frames.append(blank_frame.copy())
                mosaic_row = cv2.hconcat(row_frames)
                mosaic_rows.append(mosaic_row)

            mosaic = cv2.vconcat(mosaic_rows)
            h, w, ch = mosaic.shape

            if self.is_recording:
                mosaic = cv2.rectangle(mosaic, (0, 0), (w - 1, h - 1),
                                       (0, 0, 255), 8)

            mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
            qt_image = QtGui.QImage(
                mosaic.tobytes(), w, h, ch * w, QtGui.QImage.Format_RGB888
            )
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_recording()
        for cam in self.webcams:
            cam.release()
        event.accept()