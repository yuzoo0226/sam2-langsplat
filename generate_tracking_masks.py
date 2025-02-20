import os
import sys
import cv2
import glob
import copy
import tqdm
import time
import random
import argparse
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from PIL import Image
from icecream import ic
import threading

import torch

# from segment_anything import SamPredictor, sam_model_registry
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.grid_generator import generate_uniform_grid, draw_points_on_image, auto_mask_generator
from utils.mask_utils import apply_mask_to_rgba_image, apply_mask_to_image, filter_overlapping_masks, combined_all_mask, has_same_mask, calculate_overlap_ratio, visualize_instance
from utils.clip_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig
from utils.gpt_utils import OpenAIUtils
from clip_utils import get_features_from_image_and_masks
from segment_anything_langsplat import sam_model_registry

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,
    QSlider, QLineEdit, QComboBox, QWidget
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt


# class TrackingViewer(QMainWindow):
class TrackingViewer():
    def __init__(self, image_dir: str, sam_checkpoint: str, sam_config: str, based_gui="dpg", is_viewer=True, is_debug=True, keyframe_interval: int = 0, frame_number: bool = False, feature_type="each") -> None:
        """A GUI application for viewing sorted images with navigation support."""
        super().__init__()
        self.image_dir = image_dir
        self.video_path = None
        self.images = self._load_images(ext="jpg")
        h, w = cv2.imread(self.images[0], 0).shape
        # self.main_image_size = max(h, w)
        # self.sub_image_size = max(h, w) // 2

        self.main_image_width = w
        self.main_image_height = h

        self.show_subimage_scale = 4
        self.sub_image_width = w // self.show_subimage_scale
        self.sub_image_height = h // self.show_subimage_scale

        self.num_images = len(self.images)
        self.mask_indices = [1]
        self.target_frame_idx: int = 0
        self.target_mask_idx: int = -1
        self.ann_obj_id: int = 0
        self.ann_frame_idx = 0
        self.keyframe_interval: int = keyframe_interval
        self.video_segments = {}  # video_segments contains the per-frame segmentation results

        self.checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_ckpt_path = "./checkpoints/sam_vit_h_4b8939.pth"
        self.predictor_video = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        # self.predictor_image = build_sam2(self.model_cfg, self.checkpoint)
        self.sam_model = sam_model_registry["vit_h"](checkpoint=self.sam_ckpt_path).to('cuda')
        self.clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.inference_state = self.predictor_video.init_state(self.image_dir)
            self.frame_names = [
                p for p in os.listdir(self.image_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        self.is_viewer = is_viewer
        self.is_debug = is_debug
        self.based_gui = based_gui
        self.with_frame_number = frame_number
        self.visualization_mode = "mask"
        self.mask_level = "large"
        self.feature_type = feature_type
        self.openai_utils = OpenAIUtils()

        if is_viewer:
            if based_gui == "dpg":
                self._setup_gui()
            else:
                self.initUI()

    def _load_images(self, ext="jpg") -> Dict[int, str]:
        """Load and sort images from the specified folder.
        Args:
            ext (str): The file extension of the images.
        Returns:
            Dict[int, str]: A dictionary containing the image index and path.
        """
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, f"*.{ext}")))
        return {i: path for i, path in enumerate(image_paths)}

    def _load_texture(self, image_path: str) -> Tuple[int, int, int, np.ndarray]:
        """Load an image and convert it to a texture format.
        Args:
            image_path (str): The path to the image file.
        Returns:
            Tuple[int, int, int, np.ndarray]: The width, height, number of channels, and image data.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        # image = np.flip(image, axis=0)
        height, width, channels = image.shape
        # image_data = image.flatten() / 255.0
        return width, height, channels, image / 255.0

    @staticmethod
    def _create_video_from_folder(image_dir: str, ext="jpg", fps=10, is_gif=False) -> str:
        """Create a video from images in the specified folder.
        Args:
            image_dir (str): The directory containing the images.
            ext (str): The file extension of the images.
            fps (int): The frame rate of the video.
            is_gif (bool): Whether to create a GIF file.
        Returns:
            str: The path to the created video file.
        """

        output_path = os.path.join(image_dir, "video.mp4")
        image_files = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))

        if not image_files:
            print(f"No .{ext} images found in the directory.")
            exit()

        frame = cv2.imread(image_files[0])
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        video.write(frame)

        if is_gif:
            try:
                import imageio
                gif_images = []
                output_path_gif = output_path.replace(".m4v", ".gif")

                for image_file in image_files:
                    frame = cv2.imread(image_file)
                    gif_images.append(imageio.v3.imread(image_file))
                imageio.mimsave(output_path_gif, gif_images, duration=float(1/fps))
            except ImportError as e:
                print("You must install imageio following command. ``pip install imageio``")
                is_gif = False

        video.release()
        print(f"Video saved as {output_path}")

        if is_gif:
            print(f"GIF saved as {output_path_gif}")

        return output_path

    @staticmethod
    def _rename_folder(image_dir: str, ext="png") -> None:

        image_files = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))

        if not image_files:
            print(f"No .{ext} images found in the directory.")
            exit()

        for image_file in image_files:
            frame = cv2.imread(image_file)
            output_dir, filename = os.path.split(image_file)
            output_dir = output_dir.replace("images", "renamed_images")
            os.makedirs(output_dir, exist_ok=True)
            filename = filename.replace("frame_", "")

            output_path = os.path.join(output_dir, filename)
            print(output_path)
            cv2.imwrite(output_path, frame)

    def resize_sub_image(self, image):
        h, w = image.shape[:2]
        return cv2.resize(image, (w//self.show_subimage_scale, h//self.show_subimage_scale), interpolation=cv2.INTER_AREA)

    def set_before1_image(self, image):
        image = cv2.resize(image, (int(self.main_image_width * (1/self.show_subimage_scale)), int(self.main_image_height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
        dpg.set_value(self.before1_image_texture, image)
        dpg.configure_item(self.before1_image_texture)

    def set_before2_image(self, image):
        image = cv2.resize(image, (int(self.main_image_width * (1/self.show_subimage_scale)), int(self.main_image_height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
        dpg.set_value(self.before2_image_texture, image)
        dpg.configure_item(self.before2_image_texture)

    def set_next1_image(self, image):
        image = cv2.resize(image, (int(self.main_image_width * (1/self.show_subimage_scale)), int(self.main_image_height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
        dpg.set_value(self.next1_image_texture, image)
        dpg.configure_item(self.next1_image_texture)

    def set_next2_image(self, image):
        image = cv2.resize(image, (int(self.main_image_width * (1/self.show_subimage_scale)), int(self.main_image_height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
        dpg.set_value(self.next2_image_texture, image)
        dpg.configure_item(self.next2_image_texture)

    def set_main_image(self, image):
        dpg.set_value(self.select_image_texture, image)
        dpg.configure_item(self.select_image_texture)

    def create_video_callback(self, sender: str = None, app_data: str = None) -> None:
        """create video from selected folder

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        self.video_path = self._create_video_from_folder(self.image_dir)

    def rename_folder_callback(self, sender: str = None, app_data: str = None) -> None:
        """create video from selected folder

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        self._rename_folder(self.image_dir)

    def update_images(self, sender: int = None, app_data: int = None) -> None:
        """Update displayed images based on slider value."""
        if self.based_gui == "dpg":
            idx = int(app_data)
        else:
            idx = int(sender)
            self.slider_value_display.setText(str(idx))

        self.target_frame_idx = idx
        self.ann_frame_idx = idx

        try:
            ic(self.images[idx])
        except (IndexError, KeyError) as e:
            ic(e)
            return

        if self.target_mask_idx == -1:
            if idx in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx])
                if self.based_gui == "dpg":
                    dpg.set_value(self.select_image_texture, image_data)
                    dpg.configure_item(self.select_image_texture)
                else:
                    image = cv2.imread(self.images[idx])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    self.set_image(self.label_selected, image)

            if idx - 2 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx - 2])
                if self.based_gui == "dpg":
                    resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    dpg.set_value(self.before2_image_texture, resized_image)
                    dpg.configure_item(self.before2_image_texture)
                else:
                    image = cv2.imread(self.images[idx-2])
                    image = cv2.resize(image, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    self.set_image(self.label_before2, image)
            else:
                if self.based_gui == "dpg":
                    dpg.set_value(self.before2_image_texture, self.blank_sub_image)
                else:
                    self.set_image(self.label_before2, self.blank_sub_image)

            if idx - 1 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx - 1])
                if self.based_gui == "dpg":
                    resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    dpg.set_value(self.before1_image_texture, resized_image)
                    dpg.configure_item(self.before1_image_texture)
                else:
                    image = cv2.imread(self.images[idx-1])
                    image = cv2.resize(image, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    self.set_image(self.label_before1, image)
            else:
                if self.based_gui == "dpg":
                    dpg.set_value(self.before1_image_texture, self.blank_sub_image)
                else:
                    self.set_image(self.label_before1, self.blank_sub_image)

            if idx + 1 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx + 1])
                if self.based_gui == "dpg":
                    resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    dpg.set_value(self.next1_image_texture, resized_image)
                    dpg.configure_item(self.next1_image_texture, width=width//2, height=height//2)
                else:
                    image = cv2.imread(self.images[idx+1])
                    image = cv2.resize(image, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    self.set_image(self.label_next1, image)
            else:
                if self.based_gui == "dpg":
                    dpg.set_value(self.next1_image_texture, self.blank_sub_image)
                else:
                    self.set_image(self.label_next1, self.blank_sub_image)

            if idx + 2 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx + 2])
                if self.based_gui == "dpg":
                    resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    dpg.set_value(self.next2_image_texture, resized_image)
                    dpg.configure_item(self.next2_image_texture, width=width//2, height=height//2)
                else:
                    image = cv2.imread(self.images[idx+2])
                    image = cv2.resize(image, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
                    self.set_image(self.label_next2, image)
            else:
                if self.based_gui == "dpg":
                    dpg.set_value(self.next2_image_texture, self.blank_sub_image)
                else:
                    self.set_image(self.label_next2, self.blank_sub_image)
        else:
            self.mask_select_callback(str(self.target_mask_idx))

    def mask_to_numpy(self, mask, image, obj_id=None, random_color=False, visualize=False, filename=None, anns_obj_id=None, borders=True, use_rgb=False):
        if use_rgb:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                color = np.array([*cmap(cmap_idx)[:3], 0.6])
            h, w = mask.shape[-2:]

            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        else:
            if random_color:
                gray_value = np.random.random()
                color = np.array([gray_value, 0.6])
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                gray_value = np.mean(cmap(cmap_idx)[:3])
                color = np.array([gray_value, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color[0]

        if borders:
            contours, _ = cv2.findContours(mask_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(mask_image, contours, -1, (0, 0, 1, 0.4), thickness=1)

        if visualize:
            cv2.imshow("mask image", mask_image)
            cv2.waitKey(0)
        if filename is not None:
            # ic(os.path.join(self.image_dir, filename).replace(".", f"_{str(anns_obj_id).zfill(5)}."))
            cv2.imwrite(os.path.join(self.image_dir, filename).replace("renamed_images", "renamed_images_anns").replace(".", f"_{str(anns_obj_id).zfill(5)}."), mask_image*255)

        # if self.is_viewer:
        #     self.set_main_image(mask_image)
        return mask_image

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def change_mask_level(self, sender: str = None, app_data: str = None) -> None:
        """change mask level
        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        if self.based_gui == "dpg":
            self.mask_level = app_data
        else:
            levels = ["large", "medium", "small"]
            self.mask_level = levels[sender]

        ic(self.mask_level)

    def mask_point_generation(self, ann_frame_idx, ann_obj_id, height, width, num_points=128, image=None):
        ic(self.mask_level)

        points, segmentation_array, anns = auto_mask_generator(
            image,
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.85,
            box_nms_thresh=0.85,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            base="sam",
            mask_level=self.mask_level
        )

        self.show_anns(anns=anns)

        keys = [i for i in range(len(points))]
        _, removed_indices = filter_overlapping_masks(segmentation_array, keys, iou_threshold=0.5, del_method="smaller")

        for idx in removed_indices:
            np.delete(segmentation_array, idx)

        if self.video_segments != {}:
            # masks = self.video_segments[self.ann_frame_idx].values()
            keys = self.video_segments[self.ann_frame_idx].keys()
            ic(keys)
            combined_mask = combined_all_mask(self.video_segments[self.ann_frame_idx])

            for idx, target_mask in enumerate(segmentation_array):
                same_mask_idx = has_same_mask(target_mask, self.video_segments[self.ann_frame_idx])
                if same_mask_idx > 0:
                    removed_indices.append(idx)

                    ic(f"[Add mask] append mask {idx} in frame_{self.ann_frame_idx}.jpg as obj_id: {same_mask_idx}")
                    _, out_obj_ids, out_mask_logits = self.predictor_video.add_new_mask(
                        inference_state=self.inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=same_mask_idx,
                        mask=target_mask
                    )

                overlap_ratio = calculate_overlap_ratio(combined_mask, target_mask)
                # ic(idx, overlap_ratio)
                if overlap_ratio > 0.9:
                    removed_indices.append(idx)

        append_obj_id = 1
        for idx, mask in enumerate(segmentation_array):
            if idx in removed_indices:
                continue

            ic(f"[New obj] append mask {idx} in frame_{self.ann_frame_idx}.jpg as obj_id: {ann_obj_id+append_obj_id}")
            _, out_obj_ids, out_mask_logits = self.predictor_video.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id+append_obj_id,
                mask=mask
            )
            append_obj_id += 1

        if image is not None and self.is_viewer and self.is_debug:
            for mask in out_mask_logits:
                image = draw_points_on_image(image, points=points)
                mask = (mask > 0.0).cpu().numpy()
                display_image = apply_mask_to_rgba_image(image, mask)
                if self.based_gui == "dpg":
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2RGBA)
                    self.set_main_image(display_image/255)
                else:
                    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGRA)
                    self.set_image(self.label_selected, display_image)
                if self.is_debug:
                    # time.sleep(0.2)
                    input("show next mask >>>")

        return out_obj_ids, out_mask_logits

    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

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
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        cv2.imwrite("anns.jpg", img*255)

        if self.is_viewer:
            self.set_image(self.label_selected, img)
            # dpg.set_value(self.select_image_texture, img)
            # dpg.configure_item(self.select_image_texture)

    def del_inference_state_image(self):
        self.inference_state["images"] = self.inference_state["images"][1:]
        self.inference_state["num_frames"] -= 1
        # self.frame_names = self.frame_names[1:]

    def reset_mask_callback(self, sender: str = None, app_data: str = None) -> None:
        """reset target mask idx and show unmasked image

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        self.target_mask_idx = -1
        self.update_images(str(self.target_frame_idx), str(self.target_frame_idx))

    def mask_select_callback(self, sender: int = None, app_data: int = None) -> None:
        """mask selection

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        if self.based_gui == "dpg":
            ic(f"Selected Item: {app_data}")
            self.target_mask_idx: int = int(app_data)
        else:
            ic(f"Selected Item: {sender}")
            self.target_mask_idx: int = int(sender)

        target_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx]))
        if self.target_mask_idx > 0:
            if self.based_gui == "dpg":
                target_mask = self.video_segments[self.target_frame_idx][self.target_mask_idx].squeeze()
            else:
                try:
                    target_mask = self.video_segments[self.target_frame_idx][self.target_mask_idx].squeeze()
                except Exception as e:
                    target_mask = self.blank_main_image

            try:
                before1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-1]))
                before1_mask = self.video_segments[self.target_frame_idx-1][self.target_mask_idx].squeeze()
                before1_masked_image = apply_mask_to_image(before1_image, before1_mask)
                before1_masked_image = cv2.cvtColor(before1_masked_image, cv2.COLOR_BGR2RGBA)
                if self.based_gui == "dpg":
                    self.set_before1_image(before1_masked_image/255)
                else:
                    self.set_image(self.label_before1, before1_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                    self.set_before1_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_before1, self.blank_sub_image)

            try:
                before2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-2]))
                before2_mask = self.video_segments[self.target_frame_idx-2][self.target_mask_idx].squeeze()
                before2_masked_image = apply_mask_to_image(before2_image, before2_mask)
                before2_masked_image = cv2.cvtColor(before2_masked_image, cv2.COLOR_BGR2RGBA)
                if self.based_gui == "dpg":
                    self.set_before2_image(before2_masked_image/255)
                else:
                    self.set_image(self.label_before2, before2_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                    self.set_before2_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_before2, self.blank_sub_image)

            try:
                next1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+1]))
                next1_mask = self.video_segments[self.target_frame_idx+1][self.target_mask_idx].squeeze()
                next1_masked_image = apply_mask_to_image(next1_image, next1_mask)
                next1_masked_image = cv2.cvtColor(next1_masked_image, cv2.COLOR_BGR2RGBA)
                if self.based_gui == "dpg":
                    self.set_next1_image(next1_masked_image/255)
                else:
                    self.set_image(self.label_next1, next1_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                    self.set_next1_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_next1, self.blank_sub_image)

            try:
                next2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+2]))
                next2_mask = self.video_segments[self.target_frame_idx+2][self.target_mask_idx].squeeze()
                next2_masked_image = apply_mask_to_image(next2_image, next2_mask)
                next2_masked_image = cv2.cvtColor(next2_masked_image, cv2.COLOR_BGR2RGBA)
                if self.based_gui == "dpg":
                    self.set_next2_image(next2_masked_image/255)
                else:
                    self.set_image(self.label_next2, next2_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                    self.set_next2_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_next2, self.blank_sub_image)

        else:
            if self.based_gui == "dpg":
                if self.visualization_mode == "mask":
                    target_mask = combined_all_mask(self.video_segments[self.target_frame_idx])
                else:
                    masked_image = visualize_instance(self.video_segments[self.target_frame_idx])
            else:
                try:
                    if self.visualization_mode == "mask":
                        target_mask = combined_all_mask(self.video_segments[self.target_frame_idx])
                    else:
                        masked_image = visualize_instance(self.video_segments[self.target_frame_idx])
                except Exception as e:
                    target_mask = self.blank_main_image
            try:
                if self.visualization_mode == "mask":
                    before1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-1]))
                    before1_mask = combined_all_mask(self.video_segments[self.target_frame_idx-1])
                    before1_masked_image = apply_mask_to_image(before1_image, before1_mask)
                    before1_masked_image = cv2.cvtColor(before1_masked_image, cv2.COLOR_BGR2RGBA)
                else:
                    before1_masked_image = visualize_instance(self.video_segments[self.target_frame_idx-1])
                if self.based_gui == "dpg":
                    self.set_before1_image(before1_masked_image/255)
                else:
                    self.set_image(self.label_before1, before1_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                    self.set_before1_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_before1, self.blank_sub_image)

            try:
                if self.visualization_mode == "mask":
                    before2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-2]))
                    before2_mask = combined_all_mask(self.video_segments[self.target_frame_idx-2])
                    before2_masked_image = apply_mask_to_image(before2_image, before2_mask)
                    before2_masked_image = cv2.cvtColor(before2_masked_image, cv2.COLOR_BGR2RGBA)
                else:
                    before2_masked_image = visualize_instance(self.video_segments[self.target_frame_idx-2])
                if self.based_gui == "dpg":
                    self.set_before2_image(before2_masked_image/255)
                self.set_image(self.label_before2, before2_masked_image, auto_resize=True)
            except Exception as e:
                self.set_image(self.label_before2, self.blank_sub_image)
                # self.set_before2_image(self.blank_sub_image)

            try:
                if self.visualization_mode == "mask":
                    next1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+1]))
                    next1_mask = combined_all_mask(self.video_segments[self.target_frame_idx+1])
                    next1_masked_image = apply_mask_to_image(next1_image, next1_mask)
                    next1_masked_image = cv2.cvtColor(next1_masked_image, cv2.COLOR_BGR2RGBA)
                else:
                    next1_masked_image = visualize_instance(self.video_segments[self.target_frame_idx+1])
                if self.based_gui == "dpg":
                    self.set_next1_image(next1_masked_image/255)
                else:
                    self.set_image(self.label_next1, next1_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                    self.set_next1_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_next1, self.blank_sub_image)

            try:
                if self.visualization_mode == "mask":
                    next2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+2]))
                    next2_mask = combined_all_mask(self.video_segments[self.target_frame_idx+2])
                    next2_masked_image = apply_mask_to_image(next2_image, next2_mask)
                    next2_masked_image = cv2.cvtColor(next2_masked_image, cv2.COLOR_BGR2RGBA)
                else:
                    next2_masked_image = visualize_instance(self.video_segments[self.target_frame_idx+2])
                if self.based_gui == "dpg":
                    self.set_next2_image(next2_masked_image/255)
                else:
                    self.set_image(self.label_next2, next2_masked_image, auto_resize=True)
            except Exception as e:
                if self.based_gui == "dpg":
                   self.set_next2_image(self.blank_sub_image)
                else:
                    self.set_image(self.label_next2, self.blank_sub_image)
        try:
            if self.visualization_mode == "mask":
                masked_image = apply_mask_to_image(target_image, target_mask)
                masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGBA)
            if self.based_gui == "dpg":
                self.set_main_image(masked_image/255)
            else:
                self.set_image(self.label_selected, masked_image)
        except Exception as e:
            ic(e)

    def add_mask_to_blank_area(self, sender: str = None, app_data: str = None) -> None:
        """_summary_

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        target_mask = combined_all_mask(self.video_segments[self.target_frame_idx])
        target_mask = (~target_mask).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(target_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500

        color_mask = cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR)

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            ic(idx, area)
            if area >= min_area:
                cv2.drawContours(color_mask, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # area
                cv2.drawContours(color_mask, [contour], -1, (0, 0, 255), thickness=2)  # border

        visualized_image = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGBA)
        self.set_main_image(visualized_image/255)

    def change_target_frame_idx(self, idx: int) -> None:
        """Change the target frame index and update the displayed images.
        Args:
            idx (int): The new target frame index.
        """
        self.target_frame_idx = idx
        self.ann_frame_idx = idx

    def dump_video_callback(self, sender: str = None, app_data: str = None) -> None:
        """dump gif each obj_id

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        try:
            import imageio

        except ModuleNotFoundError as e:
            ic(e)
            ic("Please Install imageio with following commands, ``pip install imageio''")
            return False

        for obj_id in tqdm.tqdm(range(1, len(self.video_segments[self.ann_frame_idx]))):
            output_path = f"./language_features/output_obj_{str(obj_id).zfill(3)}.gif"
            gif_data = []
            mp4_data = []
            for frame_id, masks in self.video_segments.items():
                if obj_id not in masks:
                    current_mask = np.zeros((self.main_image_height, self.main_image_width, 1), dtype=np.uint8)
                else:
                    current_mask = masks[obj_id].squeeze()
                image = cv2.imread(self.images[frame_id])
                current_bgr = apply_mask_to_image(image, current_mask)
                if self.with_frame_number:
                    current_bgr = cv2.putText(current_bgr, f"frame_{frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                mp4_data.append(current_bgr)
                gif_data.append(cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB))

            # gif
            imageio.mimsave(output_path, gif_data, fps=2)

            height, width, _ = mp4_data[0].shape
            output_path = output_path.replace(".gif", ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 1, (width, height))

            # mp4
            for img in mp4_data:
                video_writer.write(img)

            video_writer.release()

        output_path = "./language_features/output_instance_visualizer.mp4"
        video_writer_instance = cv2.VideoWriter(output_path, fourcc, 1, (width*2, height))
        for idx in range(len(self.frame_names)):
            ic(idx)
            image_visualize_instance = visualize_instance(self.video_segments[idx])
            # try:
            image_visualize_instance = cv2.cvtColor(image_visualize_instance, cv2.COLOR_RGBA2BGR)
            image_orig_bgr = cv2.imread(self.images[idx])
            combined_image = cv2.hconcat([image_orig_bgr, image_visualize_instance])
            combined_image = cv2.putText(combined_image, f"frame_{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            video_writer_instance.write(combined_image)

        video_writer_instance.release()
        ic("Video writing completed.")

        return

    def dump_mask_npy_callback(self, sender: str = None, app_data: str = None) -> None:
        """Dump Mask as Numpy Array

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        for frame_id, masks in tqdm.tqdm(self.video_segments.items()):
            output_path = f"./language_features/frame_{str(frame_id+1).zfill(5)}_s.npy"
            height, width = next(iter(masks.values())).shape[1:]
            output_array = np.full((4, height, width), -1, dtype=int)

            # Save masks in lists
            for label, mask in masks.items():
                output_array[3][mask[0]] = label - 1  # obj_idが1から始まるため

            ic(output_path)

            np.save(output_path, np.array(output_array, dtype=np.float32))

    def dump_feature_callback(self, sender: str = None, app_data: str = None) -> None:
        """Dump CLIP Features

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        output_path = "language_features/features.npy"
        self.clip_model.eval()
        region_features = []
        each_obj_infos = {}
        ic(self.ann_frame_idx)
        ic(self.video_segments[0])
        ic(self.video_segments[self.ann_frame_idx])

        # for obj_id in tqdm.tqdm(range(0, len(self.video_segments[0]))):
        for obj_id, item in tqdm.tqdm(self.video_segments[self.ann_frame_idx].items()):
            if obj_id not in each_obj_infos:
                each_obj_infos[obj_id] = {"max_area": 0, "target_frame": 0, "target_mask": None, "target_mask_torch": None, "region_features": None}

            for frame_id, masks in self.video_segments.items():
                try:
                    area = torch.from_numpy(masks[obj_id]).sum().item()
                    if area > each_obj_infos[obj_id]["max_area"]:
                        each_obj_infos[obj_id]["max_area"] = area
                        each_obj_infos[obj_id]["target_frame"] = frame_id
                        each_obj_infos[obj_id]["target_mask"] = masks[obj_id].squeeze()
                        each_obj_infos[obj_id]["target_mask_torch"] = masks[obj_id]
                except Exception as e:
                    ic(e)
                    ic(f"No segmentation results about obj_id: {obj_id} in frame_{frame_id}")

        ic(each_obj_infos)

        for obj_id, obj_info in tqdm.tqdm(each_obj_infos.items()):
            target_frame = obj_info["target_frame"]
            target_mask = obj_info["target_mask"]
            target_mask_torch = obj_info["target_mask_torch"]

            image = cv2.imread(self.images[target_frame])
            masked_bgr = apply_mask_to_image(image, target_mask)
            masked_rgb = apply_mask_to_image(image, target_mask, convert_to_rgb=True, is_crop=True)
            if self.is_viewer:
                visualized_image = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGBA)
                if self.based_gui == "dpg":
                    self.set_main_image(visualized_image/255)
                else:
                    self.set_image(self.label_selected, visualized_image)

            with torch.no_grad():
                tile_tensor = torch.from_numpy(masked_rgb).to(torch.float32)
                tile_tensor = tile_tensor.permute(2, 0, 1)
                tile_tensor = tile_tensor / 255.0
                tiles = tile_tensor.unsqueeze(0)
                tiles = tiles.to("cuda")
                clip_features = self.clip_model.encode_image(tiles)
                clip_features /= clip_features.norm(dim=-1, keepdim=True)
                # clip_features = clip_features.detach().cpu().half()
                clip_features = clip_features.detach().to("cpu")
                each_obj_infos[obj_id]["region_feature"] = clip_features
            # each_obj_infos[obj_id]["region_feature"] = get_features_from_image_and_masks(self.clip_model, image, torch.from_numpy(target_mask_torch), background=0.)
            ic(obj_id, target_frame, obj_info["max_area"], each_obj_infos[obj_id]["region_feature"])
            # region_features.append(region_feature.squeeze())

        region_features = [obj_info["region_feature"].squeeze() for obj_info in each_obj_infos.values()]
        ic(region_features)

        np.save(output_path, np.array(region_features, dtype=np.float32))

        if self.feature_type == "union":
            ic("dump feature union masks")
            for frame_id, masks in tqdm.tqdm(self.video_segments.items()):
                output_path = f"./language_features/frame_{str(frame_id+1).zfill(5)}_f.npy"
                np.save(output_path, np.array(region_features, dtype=np.float32))
        else:
            ic("dump feature each masks")
            for frame_id, masks in tqdm.tqdm(self.video_segments.items()):
                region_features = []
                output_path = f"./language_features/frame_{str(frame_id+1).zfill(5)}_f.npy"
                image = cv2.imread(self.images[frame_id])
                for obj_id, mask in masks.items():
                    if mask.sum() == 0:
                        # ic("This mask is all False", frame_id, obj_id)
                        # feature = torch.full((512,), 0, dtype=torch.float32)
                        continue
                    else:
                        try:
                            with torch.no_grad():
                                masked_rgb = apply_mask_to_image(image, mask.squeeze(), convert_to_rgb=True, is_crop=True)
                                tile_tensor = torch.from_numpy(masked_rgb).to(torch.float32)
                                tile_tensor = tile_tensor.permute(2, 0, 1)
                                tile_tensor = tile_tensor / 255.0
                                tiles = tile_tensor.unsqueeze(0)
                                tiles = tiles.to("cuda")
                                clip_features = self.clip_model.encode_image(tiles)
                                clip_features /= clip_features.norm(dim=-1, keepdim=True)
                                clip_features = clip_features.detach().to("cpu")
                                # clip_features = clip_features.detach().cpu().half()
                                # feature = get_features_from_image_and_masks(self.clip_model, image, torch.from_numpy(mask), background=0.)
                                feature = clip_features[0]
                        except Exception as e:
                            ic(e)
                            ic(f"cannot get features in frame_{frame_id+1}'s obj_id: {obj_id}")
                            self.video_segments[frame_id][obj_id] = np.zeros_like(self.video_segments[frame_id][obj_id], dtype=bool)
                            # feature = torch.full((512,), 0, dtype=torch.float32)
                            continue

                    # ic(feature)
                    # ic(feature.shape)
                    feature = torch.cat((torch.tensor([obj_id], dtype=torch.float32), feature))  # 先頭にobj_idを付与
                    region_features.append(feature)

                region_features = np.array(region_features, dtype=np.float32)
                region_features = region_features.reshape(-1, 513)
                np.save(output_path, region_features)
                input(">>>")
        return

    def dump_text_feature_callback(self, sender: str = None, app_data: str = None) -> None:
        """_summary_

        Args:
            sender (str, optional): _description_. Defaults to None.
            app_data (str, optional): _description_. Defaults to None.
        """
        self.clip_model.eval()
        region_features = []
        each_obj_infos = {}
        ic(self.ann_frame_idx)
        ic(self.video_segments[0])
        ic(self.video_segments[self.ann_frame_idx])
        self.openai_utils._reset_prompt()

        # for i in range(3):
        #     choose_image = cv2.imread(random.choice(self.images))
        #     self.openai_utils._add_image_prompt(choose_image)

        # self.openai_utils._add_text_prompt("This 3d space contains the objects shown in the previous image.. When labeling the objects, give them unique characteristics that are not shared with other objects.")

        # for obj_id in tqdm.tqdm(range(0, len(self.video_segments[0]))):
        for obj_id, item in tqdm.tqdm(self.video_segments[self.ann_frame_idx].items()):
            if obj_id not in each_obj_infos:
                each_obj_infos[obj_id] = {"max_area": 0, "target_frame": 0, "target_mask": None, "target_mask_torch": None, "labels": None}

            for frame_id, masks in self.video_segments.items():
                try:
                    area = torch.from_numpy(masks[obj_id]).sum().item()
                    if area > each_obj_infos[obj_id]["max_area"]:
                        each_obj_infos[obj_id]["max_area"] = area
                        each_obj_infos[obj_id]["target_frame"] = frame_id
                        each_obj_infos[obj_id]["target_mask"] = masks[obj_id].squeeze()
                        each_obj_infos[obj_id]["target_mask_torch"] = masks[obj_id]
                except Exception as e:
                    ic(e)
                    ic(f"No segmentation results about obj_id: {obj_id} in frame_{frame_id}")

        ic(each_obj_infos)

        for obj_id, obj_info in tqdm.tqdm(each_obj_infos.items()):
            target_frame = obj_info["target_frame"]
            target_mask = obj_info["target_mask"]
            target_mask_torch = obj_info["target_mask_torch"]

            image = cv2.imread(self.images[target_frame])
            masked_bgr = apply_mask_to_image(image, target_mask)
            if self.is_viewer:
                visualized_image = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGBA)
                if self.based_gui == "dpg":
                    self.set_main_image(visualized_image/255)
                else:
                    self.set_image(self.label_selected, visualized_image)

            masked_bgr = apply_mask_to_image(image, target_mask)
            each_obj_infos[obj_id]["labels"] = self.openai_utils.make_labels(masked_bgr)
            ic(obj_id, target_frame, obj_info["max_area"], each_obj_infos[obj_id]["labels"])
            text_features = self.clip_model.encode_text(each_obj_infos[obj_id]["labels"])
            text_features_numpy = text_features.detach().to("cpu").numpy()
            labels_array = np.full((text_features_numpy.shape[0], 1), obj_id, dtype=text_features_numpy.dtype)
            text_features_numpy = np.hstack((labels_array, text_features_numpy))
            ic(text_features_numpy)
            output_path = f"language_features/obj_{str(obj_id).zfill(3)}_labels.npy"
            np.save(output_path, text_features_numpy)

    def visualize_instance_callback(self, sender: str = None, app_data: str = None) -> None:
        self.visualization_mode = "instance"
        self.update_images(self.target_frame_idx, self.target_frame_idx)
        # target_image = visualize_instance(self.video_segments[self.target_frame_idx])
        # self.set_image(self.label_selected, target_image)

    def segment_whole_video(self) -> None:
        """Segment the whole video."""
        for frame_id, frame_name in enumerate(self.frame_names):
            image = cv2.imread(os.path.join(self.image_dir, frame_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out_obj_ids, out_mask_logits = self.mask_point_generation(frame_id, 0, height=self.main_image_height, width=self.main_image_width, num_points=16*16, image=image)

            for obj_idx, segment in enumerate(out_mask_logits):
                _ = self.mask_to_numpy(segment, image, filename=frame_name, anns_obj_id=obj_idx)

    def segment_video_callback(self, sender: str = None, app_data: str = None) -> None:
        """Segment for Selected video

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        ic(self.keyframe_interval)
        image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.ann_frame_idx]))
        out_obj_ids, out_mask_logits = self.mask_point_generation(self.ann_frame_idx, self.ann_obj_id, height=self.main_image_height, width=self.main_image_width, num_points=16*16, image=image)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.ann_frame_idx]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # out_obj_ids, out_mask_logits = self.mask_point_generation(self.ann_frame_idx, self.ann_obj_id, height=h, width=w, num_points=16*16, image=image_float)
            self.ann_obj_id = out_obj_ids[-1]

            if self.keyframe_interval > 0:
                ic(f"start frame idx: {max(0, self.ann_frame_idx-self.keyframe_interval)}")
                ic(self.ann_frame_idx)
                ic(self.keyframe_interval)
                ic(self.ann_frame_idx-self.keyframe_interval)
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor_video.propagate_in_video(self.inference_state, start_frame_idx=max(0, self.ann_frame_idx-self.keyframe_interval), max_frame_num_to_track=((self.keyframe_interval*3)+1)):
                    self.video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                    }
            else:
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor_video.propagate_in_video(self.inference_state):
                    self.video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                    }

            for out_frame_idx, (_, segments) in enumerate(self.video_segments.items()):
                # if out_frame_idx == len(self.frame_names)-1:
                output_filename = self.frame_names[out_frame_idx]
                image = cv2.imread(os.path.join(self.image_dir, self.frame_names[out_frame_idx]))
                for obj_idx, segment in segments.items():
                    _ = self.mask_to_numpy(segment, image, filename=output_filename, anns_obj_id=obj_idx)

            if self.is_viewer:
                self.mask_indices = []
                self.mask_indices = [0] + out_obj_ids + [999]
                # dpg.configure_item("mask_select", items=self.mask_indices)
                self.mask_select.clear()
                self.mask_select.addItems([str(i) for i in self.mask_indices])
                self.mask_select.currentIndexChanged.connect(self.mask_select_callback)

                self.target_mask_idx = 0
                self.mask_select_callback(self.target_mask_idx)

    def initUI(self):
        self.setWindowTitle("Tracking Anything Editor")
        self.setGeometry(100, 100, 1550, 1200)

        # メインウィジェット
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # メインレイアウト
        self.layout = QVBoxLayout()

        # 画像用ラベルの作成
        self.blank_main_image = np.full((self.main_image_height, self.main_image_width, 4), 100, dtype=np.uint8)
        self.blank_sub_image = np.full((self.sub_image_height, self.sub_image_width, 4), 30, dtype=np.uint8)

        self.label_selected = QLabel(self)
        self.label_before1 = QLabel(self)
        self.label_before2 = QLabel(self)
        self.label_next1 = QLabel(self)
        self.label_next2 = QLabel(self)

        self.set_image(self.label_before1, self.blank_sub_image)
        self.set_image(self.label_before2, self.blank_sub_image)
        self.set_image(self.label_next1, self.blank_sub_image)
        self.set_image(self.label_next2, self.blank_sub_image)

        # 画像の配置
        image_layout = QHBoxLayout()
        sub_image_layout = QVBoxLayout()
        sub_image_layout.addWidget(self.label_before2)
        sub_image_layout.addWidget(self.label_before1)
        sub_image_layout.addWidget(self.label_next1)
        sub_image_layout.addWidget(self.label_next2)
        image_layout.addLayout(sub_image_layout)
        image_layout.addWidget(self.label_selected)

        self.layout.addLayout(image_layout)

        # スライダー
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)  # 水平方向
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_images - 1)
        self.slider.valueChanged.connect(self.update_images)
        slider_layout.addWidget(self.slider)

        self.slider_value_display = QLineEdit()
        self.slider_value_display.setFixedWidth(50)
        self.slider_value_display.setText("0")
        self.slider_value_display.returnPressed.connect(lambda: self.update_images(int(self.slider_value_display.text())))
        slider_layout.addWidget(self.slider_value_display)

        self.layout.addLayout(slider_layout)

        # 動画パス入力
        self.video_path = QLineEdit("/mnt/home/yuga-y/usr/splat_ws/third_party/SegAnyGAussians/models/vae_v6.pt")
        self.layout.addWidget(self.video_path)

        # セグメントボタン
        self.segment_button = QPushButton("Segment Video")
        self.segment_button.clicked.connect(self.segment_video_callback)
        self.layout.addWidget(self.segment_button)

        # キーフレーム間隔入力
        # self.keyframe_interval = QLineEdit("0")
        # self.layout.addWidget(self.keyframe_interval)

        # マスク選択
        self.mask_select = QComboBox()
        self.layout.addWidget(self.mask_select)

        # マスクリセットボタン
        self.reset_mask_button = QPushButton("Reset Mask")
        self.layout.addWidget(self.reset_mask_button)
        self.reset_mask_button.clicked.connect(self.reset_mask_callback)

        # 出力ボタン
        self.dump_gif_button = QPushButton("Dump segmentation results as GIF")
        self.dump_gif_button.clicked.connect(self.dump_video_callback) 
        self.layout.addWidget(self.dump_gif_button)
        self.dump_npy_button = QPushButton("Dump segmentation results as NPY")
        self.dump_npy_button.clicked.connect(self.dump_mask_npy_callback)
        self.layout.addWidget(self.dump_npy_button)
        self.dump_feature_button = QPushButton("Dump feature")
        self.dump_feature_button.clicked.connect(self.dump_feature_callback)
        self.layout.addWidget(self.dump_feature_button)
        self.visualize_instance_button = QPushButton("Visualize instance as Color")
        self.visualize_instance_button.clicked.connect(self.visualize_instance_callback)
        self.layout.addWidget(self.visualize_instance_button)

        # Mask level selection
        self.mask_level_select = QComboBox()
        self.mask_level_select.addItems(["large", "medium", "small"])
        self.mask_level_select.currentIndexChanged.connect(self.change_mask_level)
        self.layout.addWidget(self.mask_level_select)

        self.rename_folder_button = QPushButton("Rename Folder")
        self.rename_folder_button.clicked.connect(self.rename_folder_callback)
        self.layout.addWidget(self.rename_folder_button)

        self.dump_text_feature_button = QPushButton("Get labels and text features")
        self.dump_text_feature_button.clicked.connect(self.dump_text_feature_callback)
        self.layout.addWidget(self.dump_text_feature_button)

        self.auto_segmentation_all_frame_button = QPushButton("Segmentation All Frame")
        self.auto_segmentation_all_frame_button.clicked.connect(self.auto_scenario)
        self.layout.addWidget(self.auto_segmentation_all_frame_button)

        self.central_widget.setLayout(self.layout)

    def _setup_gui(self) -> None:
        """Setup the Dear PyGUI layout."""
        dpg.create_context()
        dpg.create_viewport(title='Tracking Anything Editor', width=1550, height=1200)

        self.blank_main_image = np.full((self.main_image_height, self.main_image_width, 4), 100, dtype=np.uint8)
        self.blank_sub_image = np.full((self.sub_image_height, self.sub_image_width, 4), 30, dtype=np.uint8)

        with dpg.texture_registry():
            # self.temp = dpg.add_dynamic_texture(width=self.image_show_size, height=self.image_show_size, default_value=blank_image, tag="image_texture")
            self.select_image_texture = dpg.add_dynamic_texture(width=self.main_image_width, height=self.main_image_height, default_value=self.blank_main_image, tag="selected_image_texture")
            self.before1_image_texture = dpg.add_dynamic_texture(width=self.sub_image_width, height=self.sub_image_height, default_value=self.blank_sub_image, tag="before1_image_texture")
            self.before2_image_texture = dpg.add_dynamic_texture(width=self.sub_image_width, height=self.sub_image_height, default_value=self.blank_sub_image, tag="before2_image_texture")
            self.next1_image_texture = dpg.add_dynamic_texture(width=self.sub_image_width, height=self.sub_image_height, default_value=self.blank_sub_image,  tag="next1_image_texture")
            self.next2_image_texture = dpg.add_dynamic_texture(width=self.sub_image_width, height=self.sub_image_height, default_value=self.blank_sub_image,  tag="next2_image_texture")

        with dpg.window(label="Image Processing GUI"):
            # self.text_current_frame_widget_id = dpg.add_text("frame_name", tag="frame_name")
            with dpg.group(horizontal=True):
                with dpg.group(horizontal=False):
                    dpg.add_image("before2_image_texture", label="before2 image", tag="before2_image_widget")
                    dpg.add_image("before1_image_texture", label="before1 image", tag="before1_image_widget")
                    dpg.add_image("next1_image_texture", label="next1 image", tag="next1_image_widget")
                    dpg.add_image("next2_image_texture", label="next2 image", tag="next2_image_widget")
                dpg.add_image("selected_image_texture", label="selected image", tag="select_image_widget")

            dpg.add_slider_int(label="Image ID", min_value=0, max_value=self.num_images-1, default_value=0, callback=self.update_images)
            dpg.add_input_text(label="Video path", default_value="/mnt/home/yuga-y/usr/splat_ws/third_party/SegAnyGAussians/models/vae_v6.pt", tag="checkpoint_path")

            dpg.add_button(label="Segment Video", callback=self.segment_video_callback)
            dpg.add_input_text(label="Keyframe interval", default_value="0", tag="keyframe_interval")

            # dpg.add_button(label="Create Video", callback=self.create_video_callback)
            dpg.add_button(label="Renane folder", callback=self.rename_folder_callback)

            with dpg.group(horizontal=True):
                dpg.add_combo(label="Choose an mask", items=self.mask_indices, callback=self.mask_select_callback, tag="mask_select")
                dpg.add_button(label="Reset Mask", callback=self.reset_mask_callback)

            dpg.add_combo(label="Mask Level", items=["large", "medium", "small"], callback=self.change_mask_level, default_value="large")

            # dpg.add_button(label="Add mask to target frame", callback=self.add_mask_to_blank_area)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Dump segmentation results as GIF", callback=self.dump_video_callback)
                dpg.add_button(label="Dump segmentation results as NPY", callback=self.dump_mask_npy_callback)
                dpg.add_button(label="Dump feature", callback=self.dump_feature_callback)

            # with dpg.handler_registry():
            #     dpg.add_mouse_move_handler(callback=self.mouse_move_callback)
            #     dpg.add_mouse_click_handler(callback=self.click_callback)
            #     dpg.add_mouse_release_handler(callback=self.release_callback)

        # dpg.create_viewport(title='Image Viewer', width=800, height=600)
        # with dpg.window(label="Main Window", width=800, height=600):
        #     with dpg.group(horizontal=True):
        #         dpg.add_image(self.left_texture)
        #         dpg.add_image(self.main_texture)
        #         dpg.add_image(self.right_texture)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()

    def auto_scenario(self, sender: str = None, app_data: str = None) -> None:
        """Auto Scenario
        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        # def auto_scenario_thread():
        if not os.path.exists("./language_features"):
            os.makedirs("./language_features", exist_ok=True)

        interval = self.keyframe_interval

        for idx in range(1, len(self.images), interval):
            ic(f"target_frame is {idx}")
            self.change_target_frame_idx(idx)
            self.segment_video_callback()

        # viewer.segment_whole_video()
        viewer.dump_feature_callback()
        viewer.dump_video_callback()
        viewer.dump_mask_npy_callback()
        viewer.dump_text_feature_callback()

        # thread = threading.Thread(target=auto_scenario_thread)
        # thread.start()

    def set_image(self, label, image_data, auto_resize=False):
        if auto_resize:
            image_data = self.resize_sub_image(image_data)
        height, width, channel = image_data.shape
        bytes_per_line = 4 * width
        q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking Anything GUI")
    parser.add_argument("--input_dir", "-i", default="/mnt/home/yuga-y/usr/splat_ws/datasets/shapenets/ShapeSplat2_cans_v2/renamed_images/", type=str, help="Input directory path")
    parser.add_argument("--model", "-m", default="./checkpoints/sam2.1_hiera_large.pt", type=str, help="Model checkpoint path")
    parser.add_argument("--config", "-c", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str, help="Config file path")
    parser.add_argument("--gui", "-g", default="qt", choices=["dpg", "qt"], help="GUI library")
    parser.add_argument("--feature_type", default="each", choices=["each", "union"], help="feature mask")
    parser.add_argument("--viewer_disable", "-v", action="store_false", help="Disable viewer")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    parser.add_argument("--frame_number", "-f", action="store_false", help="output video with frame number")
    parser.add_argument("--keyframe_interval", "-k", default=10, type=int, help="Keyframe interval")
    args = parser.parse_args()

    ic(args.gui)
    ic(args.viewer_disable)
    if args.viewer_disable:
        app = QApplication(sys.argv)
    viewer = TrackingViewer(args.input_dir, args.model, args.config, args.gui, args.viewer_disable, is_debug=args.debug, keyframe_interval=args.keyframe_interval, frame_number=args.frame_number, feature_type=args.feature_type)
    if args.viewer_disable:
        viewer.show()
        sys.exit(app.exec())
    else:
        viewer.auto_scenario()
