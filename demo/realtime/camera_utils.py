import cv2
from cv2_enumerate_cameras import enumerate_cameras, CameraInfo


def get_available_cameras(api_preference=cv2.CAP_ANY) -> list[CameraInfo]:
    cameras_cap: list[cv2.VideoCapture] = []
    cameras_info: list[CameraInfo] = []
    for camera_info in enumerate_cameras(api_preference):
        try:
            cap = cv2.VideoCapture(camera_info.index)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            if not ret:
                cap.release()
                continue

            cameras_cap.append(cap)
            cameras_info.append(camera_info)
        except Exception:
            # Don't consider camera if it cannot be opened or if it cannot read a frame
            continue

    for camera_cap in cameras_cap:
        camera_cap.release()

    return cameras_info