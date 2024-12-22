import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2

from utils.camera import CameraModule

import os
import warnings
from tqdm import tqdm, trange
from typing import Dict, Any

class Metric3dMDE:

    padding = [123.675, 116.28, 103.53]
    mean = np.array([123.675, 116.28, 103.53])[None, None, :]
    std = np.array([58.395, 57.12, 57.375])[None, None, :]
    model_to_input_size = {
        'vit': (616, 1064),
        'convnext': (544, 1216),
    }

    @staticmethod
    def _load_images(path_to_images: str) -> Dict[str, np.ndarray]:
        image_names = sorted(os.listdir(path_to_images))
        name_to_image = dict()
        for image_name in tqdm(image_names, leave=False, desc='Loading images'):
            image = Image.open(os.path.join(path_to_images, image_name))
            image = np.array(image)
            name_to_image[image_name.split('.')[0]] = image

        return name_to_image

    @staticmethod
    def _load_model(model_type: str) -> torch.nn.Module:
        # assert model_type in ['vit_small', 'vit_large', 'convnext_small', 'convnext_large']

        warnings.filterwarnings("ignore", category=UserWarning, module="torch.hub")
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
        model = model.eval()

        return model

    def __init__(
        self,
        path_to_camera: str,
        crop_size: int=10,
        model_type: str='vit',
    ) -> None:
        assert crop_size >= 0
        assert model_type in ['vit', 'convnext']

        self.cam_params = CameraModule(path_to_camera).get_camera_intrinsic()
        if crop_size > 0:
            self.central_crop = lambda x: x[crop_size:-crop_size, crop_size:-crop_size, :]
        else:
            self.central_crop = lambda x: x
        self.input_size = Metric3dMDE.model_to_input_size[model_type]
        self.model_type = model_type

    def preprocess_images(self, path_to_images: str) -> Dict[str, Any]:
        images_processed = {
            'name': list(),
            'image': list(),
            'pad_info': list(),
            'size': list(),
        }
        name_to_image = Metric3dMDE._load_images(path_to_images)
        for file, image in tqdm(name_to_image.items(), leave=False, desc='Preprocessing images'):
            # crop image to get rid of artifacts 
            image = self.central_crop(image)
            h, w = image.shape[:2]
            # keep ratio resize
            scale = min(self.input_size[0] / h, self.input_size[1] / w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR
            )
            # padding to input_size
            pad_h, pad_w = \
                self.input_size[0] - image.shape[0], \
                self.input_size[1] - image.shape[1]
            pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
            image = cv2.copyMakeBorder(
                image,
                pad_h_half, pad_h - pad_h_half,
                pad_w_half, pad_w - pad_w_half,
                cv2.BORDER_CONSTANT, value=Metric3dMDE.padding
            )
            pad_info = (pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half)
            # normalize
            image = np.divide((image - Metric3dMDE.mean), Metric3dMDE.std)
            image = torch.from_numpy(image).permute(2, 0, 1).type(torch.float32)
            images_processed['name'].append(file)
            images_processed['image'].append(image)
            images_processed['pad_info'].append(pad_info)
            images_processed['size'].append((h, w))

        # remember to scale intrinsic, hold depth
        focal_scaled = self.cam_params['focal_p'][0] * scale
        return images_processed, focal_scaled

    def predict_processed(self, images_processed: Dict[str, Any], focal_scaled: float) -> Dict[str, np.ndarray]:
        ##### canonical camera space #####
        # inference
        dataloader = DataLoader(
            images_processed['image'],
            shuffle=False,
            batch_size=64,
            collate_fn=lambda x: torch.cat([stack.unsqueeze(0) for stack in x], dim=0)
        )
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = Metric3dMDE._load_model(self.model_type).to(device)
        depths = list()
        for batch in tqdm(dataloader, leave=False, desc='Predicting'):
            with torch.no_grad():
                pred_depth, confidence, output_dict = \
                    model.inference({'input': batch.to(device)})
                depths.append(pred_depth)
        # saving predicted depths
        images_processed['depth'] = list(torch.cat(depths, dim=0))
        
        # 1000.0 is the focal length of canonical camera
        canonical_to_real_scale = focal_scaled / 1000.0
        final_predictions = dict()
        for i in trange(len(images_processed['depth']), leave=False, desc='Post-processing'):
            name = images_processed['name'][i]
            pred_depth = images_processed['depth'][i]
            pad_info = images_processed['pad_info'][i]
            orig_size = images_processed['size'][i]
            # unpad
            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[
                pad_info[0] : pred_depth.shape[0] - pad_info[1],
                pad_info[2] : pred_depth.shape[1] - pad_info[3]
            ]
            # upsample to original size
            pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], orig_size, mode='bilinear').squeeze()
        ##### canonical camera space #####
            # de-canonical transform / now the depth is metric
            pred_depth = pred_depth * canonical_to_real_scale
            pred_depth = pred_depth.cpu().detach().numpy()
            pred_depth = np.clip(pred_depth, 0, 10)
            final_predictions[name] = np.array(pred_depth)

        return final_predictions