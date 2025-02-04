import os
import cv2
import glob
import copy
import time
# import imageio
import argparse
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from PIL import Image
from icecream import ic

import torch

# from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class AutomaskVisualizer:
    """A GUI application for viewing sorted images with navigation support."""

    def __init__(self) -> None:
        """Initialize the image viewer with the given image folder."""
        # self.image_dir = image_dir
        # self.video_path = None
        # self.images = self._load_images(ext="jpg")
        # h, w = cv2.imread(self.images[0], 0).shape
        # self.main_image_size = max(h, w)
        # self.sub_image_size = max(h, w) // 2

        # self.main_image_width = w
        # self.main_image_height = h

        # self.show_subimage_scale = 4
        # self.sub_image_width = w // self.show_subimage_scale
        # self.sub_image_height = h // self.show_subimage_scale

        # self.num_images = len(self.images)
        # self.mask_indices = [1]
        # self.target_frame_idx: int = 0
        # self.target_mask_idx: int = -1
        # self.ann_obj_id: int = 0
        # self.ann_frame_idx = 0
        # self.video_segments = {}  # video_segments contains the per-frame segmentation results

        self.checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.model = build_sam2(self.model_cfg, self.checkpoint)

        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        #     self.inference_state = self.predictor_video.init_state(self.image_dir)
        #     self.frame_names = [
        #         p for p in os.listdir(self.image_dir)
        #         if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        #     ]
        #     self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # self.is_viewer = is_viewer
        # self.debug = is_debug
        np.random.seed(3)

    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        # ax = plt.gca()
        # ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1), thickness=1) 

        # ax.imshow(img)
        return img

    def main(self):
        image = cv2.imread("/mnt/home/yuga-y/usr/splat_ws/datasets/lerf_ovs/figurines/renamed_images_orig/00001.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_generator = SAM2AutomaticMaskGenerator(self.model)
        masks = mask_generator.generate(image)
        masked_image = self.show_anns(masks)
        ic)
        masked_image = cv2.cvtColor(masked_image*255, cv2.COLOR_RGBA2BGR)
        cv2.imwrite("mask.png", masked_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking Anything GUI")

    # parser.add_argument("--sam_checkpoint_path", default='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth', type=str)
    # parser.add_argument("--vae_checkpoint_path", default='./langsplat_variaous_autoencoder_v2', type=str)
    # parser.add_argument("--clip_model_type", default='ViT-B-16', type=str)
    # parser.add_argument("--input_dir", "-i", default="/mnt/home/yuga-y/usr/splat_ws/datasets/lerf_ovs/waldo_kitchen/renamed_images/", type=str)
    # parser.add_argument("--input_dir", "-i", default="/mnt/home/yuga-y/usr/splat_ws/datasets/lerf_ovs/figurines/renamed_images/", type=str)
    # parser.add_argument("--model", "-m", default="./checkpoints/sam2.1_hiera_large.pt", type=str)
    # parser.add_argument("--config", "-c", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    # parser.add_argument("--viewer_disable", "-v", action="store_false")
    # args = parser.parse_args()

    # viewer = TrackingViewer(args.input_dir, args.model, args.config, args.viewer_disable)
    # viewer.segment_video_callback("temp", "temp")
    cls = AutomaskVisualizer()
    cls.main()
