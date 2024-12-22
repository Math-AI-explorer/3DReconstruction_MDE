import numpy as np
import math

from typing import Dict, Tuple


class CameraModule:

    def __init__(self, path_to_camera: str) -> None:
        self.path_to_camera = path_to_camera
        self._load_camera_intrinsic()

    def _load_camera_intrinsic(self) -> None:
        with open(self.path_to_camera, "r", encoding="utf-8") as fin:
            numbers =fin\
                .read()\
                .replace('\n', ' ')\
                .strip()\
                .split(' ')
            numbers = np.array(list(map(float, numbers))).reshape(4, -1)
        fx, fy, cx, cy = numbers[0, 0], numbers[1, 1], numbers[0, 2], numbers[1, 2]

        self.camera_parameters = {
            'focal_p': (fx, fy),
            'optical_p': (cx, cy),
        }

    def save_camera_intrinsic(self, path_to_save: str) -> None:
        fx, fy = self.camera_parameters['focal_p']
        cx, cy = self.camera_parameters['optical_p']
        all_cam_params = [fx, fy, cx, cy]
        max_prec = max(list(map(lambda x: len(str(x).split('.')[1]), all_cam_params)))
        zero_num = f"0.{'0'*max_prec}"
        one_num = f"1.{'0'*max_prec}"
        with open(path_to_save, "w", encoding="utf-8") as fin:
            fin.write(f"{fx} {zero_num} {cx} {zero_num}\n")
            fin.write(f"{zero_num} {fy} {cy} {zero_num}\n")
            fin.write(f"{zero_num} {zero_num} {one_num} {zero_num}\n")
            fin.write(f"{zero_num} {zero_num} {zero_num} {zero_num}\n")

    def get_camera_intrinsic(self) -> Dict[str, Tuple[float, float]]:
        return self.camera_parameters

    def get_camera_matrix(self) -> np.ndarray:
        fx, fy = self.camera_parameters['focal_p']
        cx, cy = self.camera_parameters['optical_p']
        return np.array([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ])

    def update_camera_intrinsic(
        self, camera_parameters: Dict[str, Tuple[float, float]]
    ) -> None:
        self.camera_parameters = camera_parameters

    @staticmethod
    def convert_matrix_to_params(
        matrix: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        assert len(matrix.shape) == 2
        assert matrix.shape[0] == matrix.shape[1] == 3
        return {
            'focal_p': (matrix[0, 0], matrix[1, 1]),
            'optical_p': (matrix[0, 2], matrix[1, 2]),
        }

    @staticmethod
    def convert_params_to_matrix(
        camera_parameters: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        assert len(camera_parameters) == 2
        assert 'focal_p' in camera_parameters
        assert 'optical_p' in camera_parameters
        fx, fy = camera_parameters['focal_p']
        cx, cy = camera_parameters['optical_p']
        return np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])

    @staticmethod
    def get_adapted_matrix(intrinsic, desired_resolution, original_resolution):
        '''Get adjusted camera intrinsics.'''
        if original_resolution == desired_resolution:
            return intrinsic

        resize_width = int(math.floor(desired_resolution[1] * float(
                        original_resolution[0]) / float(original_resolution[1])))

        adapted_intrinsic = intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(original_resolution[0])
        adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(original_resolution[1])
        adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(original_resolution[0] - 1)
        adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(original_resolution[1] - 1)

        return adapted_intrinsic