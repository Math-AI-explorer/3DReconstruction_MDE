import pycolmap

import os
import shutil

from typing import Dict, Any


class SfmModule:

    def __init__(self, path_to_reconstruction: str | None = None) -> None:
        self.path_to_reconstruction = path_to_reconstruction
        self._load_reconstruction()

    def _load_reconstruction(self) -> None:
        if self.path_to_reconstruction is not None:
            self.reconstruction = pycolmap.Reconstruction(self.path_to_reconstruction)
        else:
            self.reconstruction = None

    def make_reconstruction(
        self,
        images_dir: str,
        output_path: str
    ) -> pycolmap.Reconstruction:
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path)
        database_path = os.path.join(output_path, 'database.db')

        pycolmap.extract_features(database_path, images_dir)
        pycolmap.match_exhaustive(database_path)
        maps = pycolmap.incremental_mapping(database_path, images_dir, output_path)
        self.reconstruction = maps[0]
        self.reconstruction.write(output_path)

    def get_reconstruction(self) -> pycolmap.Reconstruction:
        return self.reconstruction

    def get_images_info(self) -> Dict[str, Any]:
        assert self.reconstruction is not None
        
        image_to_params = dict()
        for image_id, image in self.reconstruction.images.items():
            # print(image.summary())
            R_t = image.cam_from_world.matrix()
            R, t = R_t[:, :3], R_t[:, 3]
            image_to_params[image.name.split('.')[0]] = {
                'image_id': image_id,
                'camera_id': image.camera_id,
                'rotation': R,
                'translation': t,
            }
        
        return image_to_params

    def get_cameras_info(self) -> Dict[str, Any]:
        assert self.reconstruction is not None
        
        camera_to_params = dict()
        for camera_id, camera in self.reconstruction.cameras.items():
            # print(camera.summary())
            fx, fy = camera.focal_length_x, camera.focal_length_y
            cx, cy = camera.principal_point_x, camera.principal_point_y
            camera_to_params[camera_id] = {
                'focal_p': (fx, fy),
                'optical_p': (cx, cy),
            }

        return camera_to_params