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
from utils.grid_generator import generate_uniform_grid, draw_points_on_image, auto_grid_generator
from utils.mask_utils import make_anns_image, apply_mask_to_rgba_image, apply_mask_to_image, calculate_iou, filter_overlapping_masks, create_combined_mask, find_enclosed_regions, get_centroids, combined_all_mask, has_same_mask


class TrackingViewer:
    """A GUI application for viewing sorted images with navigation support."""

    def __init__(self, image_dir: str, sam_checkpoint: str, sam_config: str, is_viewer=True, is_debug=True) -> None:
        """Initialize the image viewer with the given image folder."""
        self.image_dir = image_dir
        self.video_path = None
        self.images = self._load_images(ext="jpg")
        h, w = cv2.imread(self.images[0], 0).shape
        self.main_image_size = max(h, w)
        self.sub_image_size = max(h, w) // 2

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
        self.video_segments = {}  # video_segments contains the per-frame segmentation results

        self.checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor_video = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        self.predictor_image = build_sam2(self.model_cfg, self.checkpoint)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.inference_state = self.predictor_video.init_state(self.image_dir)
            self.frame_names = [
                p for p in os.listdir(self.image_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        self.is_viewer = is_viewer
        self.debug = is_debug
        if is_viewer:
            self._setup_gui()

    def _load_images(self, ext="png") -> Dict[int, str]:
        """Load and sort images from the specified folder."""
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, f"*.{ext}")))
        return {i: path for i, path in enumerate(image_paths)}

    def _load_texture(self, image_path: str) -> Tuple[int, int, int, np.ndarray]:
        """Load an image and convert it to a texture format."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        # image = np.flip(image, axis=0)
        height, width, channels = image.shape
        # image_data = image.flatten() / 255.0
        return width, height, channels, image / 255.0

    @staticmethod
    def _create_video_from_folder(image_dir: str, ext="jpg", fps=10, is_gif=False) -> str:

        output_path = os.path.join(image_dir, "video.mp4")
        image_files = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))

        if not image_files:
            print(f"No .{ext} images found in the directory.")
            exit()

        # 最初の画像を読み込み、サイズを取得
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
            except ModuleNotFoundError as e:
                print("You must install imageio following command. ``pip install imageio``")
                is_gif = False

        # MP4動画を保存
        video.release()
        print(f"Video saved as {output_path}")

        if is_gif:
            print(f"GIF saved as {output_path_gif}")

        return output_path

    @staticmethod
    def _rename_folder(image_dir: str, ext="jpg") -> None:

        image_files = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))

        if not image_files:
            print(f"No .{ext} images found in the directory.")
            exit()

        # 最初の画像を読み込み、サイズを取得
        for image_file in image_files:
            frame = cv2.imread(image_file)
            output_dir, filename = os.path.split(image_file)
            output_dir = output_dir.replace("images", "renamed_images")
            os.makedirs(output_dir, exist_ok=True)
            filename = filename.replace("frame_", "")

            output_path = os.path.join(output_dir, filename)
            print(output_path)
            cv2.imwrite(output_path, frame)

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

    def create_video_callback(self, sender: str, app_data: str) -> None:
        """create video from selected folder

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        self.video_path = self._create_video_from_folder(self.image_dir)

    def rename_folder_callback(self, sender: str, app_data: str) -> None:
        """create video from selected folder

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        self._rename_folder(self.image_dir)

    def update_images(self, sender: int, app_data: int) -> None:
        """Update displayed images based on slider value."""
        idx = int(app_data)
        self.target_frame_idx = idx
        self.ann_frame_idx = idx

        if self.target_mask_idx == -1:
            if idx in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx])
                dpg.set_value(self.select_image_texture, image_data)
                dpg.configure_item(self.select_image_texture)

            if idx - 2 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx - 2])
                resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                dpg.set_value(self.before2_image_texture, resized_image)
                dpg.configure_item(self.before2_image_texture)
            else:
                dpg.set_value(self.before2_image_texture, self.blank_sub_image)

            if idx - 1 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx - 1])
                resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                dpg.set_value(self.before1_image_texture, resized_image)
                dpg.configure_item(self.before1_image_texture)
            else:
                dpg.set_value(self.before1_image_texture, self.blank_sub_image)

            if idx + 1 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx + 1])
                resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                dpg.set_value(self.next1_image_texture, resized_image)
                dpg.configure_item(self.next1_image_texture, width=width//2, height=height//2)
            else:
                dpg.set_value(self.next1_image_texture, self.blank_sub_image)

            if idx + 2 in self.images:
                width, height, channels, image_data = self._load_texture(self.images[idx + 2])
                resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
                dpg.set_value(self.next2_image_texture, resized_image)
                dpg.configure_item(self.next2_image_texture, width=width//2, height=height//2)
            else:
                dpg.set_value(self.next2_image_texture, self.blank_sub_image)

        else:
            self.mask_select_callback("", str(self.target_mask_idx))

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
                # ランダムなグレースケール値 (0~1 の範囲)
                gray_value = np.random.random()
                color = np.array([gray_value, 0.6])  # [グレースケール値, アルファ値]
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                gray_value = np.mean(cmap(cmap_idx)[:3])  # RGB を平均してグレースケールに変換
                color = np.array([gray_value, 0.6])  # [グレースケール値, アルファ値]
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

    def mask_point_generation(self, ann_frame_idx, ann_obj_id, height, width, num_points=128, image=None):
        # points = generate_uniform_grid(height=height, width=width, num_points=num_points)
        points, segmentation_array, anns = auto_grid_generator(
            image,
            self.predictor_image,
            # min_mask_region_area=100,
            pred_iou_thresh=0.9,
            points_per_side=32,
            crop_n_points_downscale_factor=0.9,
            stability_score_thresh=0.95,
            box_nms_thresh=0.7,
            # use_m2m=True
        )

        keys = [i for i in range(len(points))]
        _, removed_indices = filter_overlapping_masks(segmentation_array, keys, del_method="smaller")

        for idx in removed_indices:
            np.delete(segmentation_array, idx)

        if self.video_segments != {}:
            masks = self.video_segments[self.ann_frame_idx].values()
            for idx, target_mask in enumerate(segmentation_array):
                if has_same_mask(target_mask, masks):
                    removed_indices.append(idx)

        append_obj_id = 1
        for idx, mask in enumerate(segmentation_array):
            if idx in removed_indices:
                continue

            _, out_obj_ids, out_mask_logits = self.predictor_video.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id+append_obj_id,
                mask=mask
            )
            append_obj_id += 1

        if image is not None:
            for mask in out_mask_logits:
                image = draw_points_on_image(image, points=points)
                mask = (mask > 0.0).cpu().numpy()
                display_image = apply_mask_to_rgba_image(image, mask)
                # display_image = cv2.cvtColor(display_image, cv2.COLOR_BGRA2RGBA)
                self.set_main_image(display_image/255)
                if self.debug:
                    time.sleep(0.2)

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

        dpg.set_value(self.select_image_texture, img)
        dpg.configure_item(self.select_image_texture)

    def del_inference_state_image(self):
        self.inference_state["images"] = self.inference_state["images"][1:]
        self.inference_state["num_frames"] -= 1
        # self.frame_names = self.frame_names[1:]

    def reset_mask_callback(self, sender: str, app_data: str) -> None:
        """reset target mask idx and show unmasked image

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        self.target_mask_idx = -1
        self.update_images("", str(self.target_frame_idx))

    def mask_select_callback(self, sender: str, app_data: str) -> None:
        """mask selection

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        ic(f"Selected Item: {app_data}")
        self.target_mask_idx: int = int(app_data)

        target_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx]))
        if self.target_mask_idx > 0:
            target_mask = self.video_segments[self.target_frame_idx][self.target_mask_idx].squeeze()

            try:
                before1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-1]))
                before1_mask = self.video_segments[self.target_frame_idx-1][self.target_mask_idx].squeeze()
                before1_masked_image = apply_mask_to_image(before1_image, before1_mask)
                before1_masked_image = cv2.cvtColor(before1_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_before1_image(before1_masked_image/255)
            except Exception as e:
                self.set_before1_image(self.blank_sub_image)

            try:
                before2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-2]))
                before2_mask = self.video_segments[self.target_frame_idx-2][self.target_mask_idx].squeeze()
                before2_masked_image = apply_mask_to_image(before2_image, before2_mask)
                before2_masked_image = cv2.cvtColor(before2_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_before2_image(before2_masked_image/255)
            except Exception as e:
                self.set_before2_image(self.blank_sub_image)

            try:
                next1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+1]))
                next1_mask = self.video_segments[self.target_frame_idx+1][self.target_mask_idx].squeeze()
                next1_masked_image = apply_mask_to_image(next1_image, next1_mask)
                next1_masked_image = cv2.cvtColor(next1_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_next1_image(next1_masked_image/255)
            except Exception as e:
                self.set_next1_image(self.blank_sub_image)

            try:
                next2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+2]))
                next2_mask = self.video_segments[self.target_frame_idx+2][self.target_mask_idx].squeeze()
                next2_masked_image = apply_mask_to_image(next2_image, next2_mask)
                next2_masked_image = cv2.cvtColor(next2_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_next2_image(next2_masked_image/255)
            except Exception as e:
                self.set_next2_image(self.blank_sub_image)

        else:
            target_mask = combined_all_mask(self.video_segments[self.target_frame_idx])

            try:
                before1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-1]))
                before1_mask = combined_all_mask(self.video_segments[self.target_frame_idx-1])
                before1_masked_image = apply_mask_to_image(before1_image, before1_mask)
                before1_masked_image = cv2.cvtColor(before1_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_before1_image(before1_masked_image/255)
            except Exception as e:
                self.set_before1_image(self.blank_sub_image)

            try:
                before2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx-2]))
                before2_mask = combined_all_mask(self.video_segments[self.target_frame_idx-2])
                before2_masked_image = apply_mask_to_image(before2_image, before2_mask)
                before2_masked_image = cv2.cvtColor(before2_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_before2_image(before2_masked_image/255)
            except Exception as e:
                self.set_before2_image(self.blank_sub_image)

            try:
                next1_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+1]))
                next1_mask = combined_all_mask(self.video_segments[self.target_frame_idx+1])
                next1_masked_image = apply_mask_to_image(next1_image, next1_mask)
                next1_masked_image = cv2.cvtColor(next1_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_next1_image(next1_masked_image/255)
            except Exception as e:
                self.set_next1_image(self.blank_sub_image)

            try:
                next2_image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.target_frame_idx+2]))
                next2_mask = combined_all_mask(self.video_segments[self.target_frame_idx+2])
                next2_masked_image = apply_mask_to_image(next2_image, next2_mask)
                next2_masked_image = cv2.cvtColor(next2_masked_image, cv2.COLOR_BGR2RGBA)
                self.set_next2_image(next2_masked_image/255)
            except Exception as e:
                self.set_next2_image(self.blank_sub_image)

        masked_image = apply_mask_to_image(target_image, target_mask)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGBA)
        self.set_main_image(masked_image/255)

    def add_mask_to_blank_area(self, sender: str, app_data: str) -> None:
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

    def segment_video_callback(self, sender: str, app_data: str) -> None:
        """Segment for Selected video

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # self.inference_state = self.predictor_video.init_state(self.image_dir)
            # self.frame_names = [
            #     p for p in os.listdir(self.image_dir)
            #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            # ]
            # self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            image = cv2.imread(os.path.join(self.image_dir, self.frame_names[self.ann_frame_idx]))
            h, w, ch = image.shape

            out_obj_ids, out_mask_logits = self.mask_point_generation(self.ann_frame_idx, self.ann_obj_id, height=h, width=w, num_points=16*16, image=image)
            self.ann_obj_id = out_obj_ids[-1]

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

            ic(out_obj_ids)
            ic(type(out_obj_ids))
            self.mask_indices = [0] + out_obj_ids
            dpg.configure_item("mask_select", items=self.mask_indices)

            self.target_mask_idx = 0
            self.mask_select_callback("segment_video_callback", self.target_mask_idx)
            # self.update_images("segment_video_callback", self.ann_frame_idx)

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
            # dpg.add_button(label="Create Video", callback=self.create_video_callback)
            # dpg.add_button(label="Renane folder", callback=self.rename_folder_callback)

            with dpg.group(horizontal=True):
                dpg.add_combo(label="Choose an mask", items=self.mask_indices, callback=self.mask_select_callback, tag="mask_select")
                dpg.add_button(label="Reset Mask", callback=self.reset_mask_callback)

            dpg.add_button(label="Add mask to target frame", callback=self.add_mask_to_blank_area)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking Anything GUI")

    # parser.add_argument("--sam_checkpoint_path", default='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth', type=str)
    # parser.add_argument("--vae_checkpoint_path", default='./langsplat_variaous_autoencoder_v2', type=str)
    # parser.add_argument("--clip_model_type", default='ViT-B-16', type=str)
    # parser.add_argument("--input_dir", "-i", default="/mnt/home/yuga-y/usr/splat_ws/datasets/lerf_ovs/waldo_kitchen/renamed_images/", type=str)
    parser.add_argument("--input_dir", "-i", default="/mnt/home/yuga-y/usr/splat_ws/datasets/lerf_ovs/figurines/renamed_images/", type=str)
    parser.add_argument("--model", "-m", default="./checkpoints/sam2.1_hiera_large.pt", type=str)
    parser.add_argument("--config", "-c", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    parser.add_argument("--viewer_disable", "-v", action="store_false")
    args = parser.parse_args()

    viewer = TrackingViewer(args.input_dir, args.model, args.config, args.viewer_disable)
    viewer.segment_video_callback("temp", "temp")
