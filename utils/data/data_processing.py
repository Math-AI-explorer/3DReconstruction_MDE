import numpy as np

from PIL import Image
import cv2
import os
import shutil


def reorganise_scene_folder(path_to_scene: str, where_save: str) -> None:
    fields = {
        'png': 'depth_maps',
        'jpg': 'images',
        'txt': 'poses',
    }

    shutil.rmtree(where_save, ignore_errors=True)
    os.makedirs(where_save)
    os.makedirs(os.path.join(where_save, 'depth_maps'))
    os.makedirs(os.path.join(where_save, 'images'))
    os.makedirs(os.path.join(where_save, 'poses'))

    for file in sorted(os.listdir(path_to_scene)):
        if os.path.isdir(file):
            continue
        file_name, file_ext = file.split('.')
        fin_dir = \
            os.path.join(where_save, fields[file_ext]) \
            if file_name.isdigit() \
            else where_save
        shutil.copy(os.path.join(path_to_scene, file), fin_dir)


def adjust_size_across_mask(
    path_to_mask: str,
    path_to_images: str,
    output_path: str,
) -> None:
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    mask_to_size = dict()
    for mask_file in sorted(os.listdir(path_to_mask)):
        path_to_file = os.path.join(path_to_mask, mask_file)
        map_h, map_w = Image.open(path_to_file).size
        mask_name = mask_file.split('.', maxsplit=1)[0]
        mask_to_size[mask_name] = (map_h, map_w)

    for image_file in sorted(os.listdir(path_to_images)):
        path_to_image = os.path.join(path_to_images, image_file)
        image = np.array(Image.open(path_to_image))
        image_name = image_file.split('.', maxsplit=1)[0]
        image = cv2.resize(image, mask_to_size[image_name])
        Image.fromarray(image).save(os.path.join(output_path, image_file))