import os
import cv2
import glob
import copy
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
from utils.grid_generator import generate_uniform_grid, draw_points_on_image
from utils.mask_utils import calculate_iou, filter_overlapping_masks, create_combined_mask, find_enclosed_regions, get_centroids


class TrackingViewer:
    """A GUI application for viewing sorted images with navigation support."""

    def __init__(self, image_dir: str, sam_checkpoint: str, sam_config: str) -> None:
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
        self.idx = 0

        # self.predictor_video = SamPredictor(model)

        # sam = sam_model_registry["vit-h"](checkpoint="<path/to/checkpoint>")
        # predictor = SamPredictor(sam)
        # predictor.set_image(<your_image>)
        # masks, _, _ = predictor.predict(<input_prompts>)

        self.checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor_video = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
        self.predictor_image = build_sam2(self.model_cfg, self.checkpoint, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(self.predictor_image)

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
        self.idx = idx
        if idx in self.images:
            width, height, channels, image_data = self._load_texture(self.images[idx])
            dpg.set_value(self.select_image_texture, image_data)
            dpg.configure_item(self.select_image_texture)

        if idx - 1 in self.images:
            width, height, channels, image_data = self._load_texture(self.images[idx - 1])

            resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
            dpg.set_value(self.before_image_texture, resized_image)
            dpg.configure_item(self.before_image_texture)
        else:
            dpg.set_value(self.before_image_texture, self.blank_sub_image)

        if idx + 1 in self.images:
            width, height, channels, image_data = self._load_texture(self.images[idx + 1])
            resized_image = cv2.resize(image_data, (int(width * (1/self.show_subimage_scale)), int(height * (1/self.show_subimage_scale))), interpolation=cv2.INTER_AREA)
            dpg.set_value(self.next_image_texture, resized_image)
            dpg.configure_item(self.next_image_texture, width=width//2, height=height//2)
        else:
            dpg.set_value(self.next_image_texture, self.blank_sub_image)

    def mask_to_numpy(self, mask, obj_id=None, random_color=False, visualize=False, filename=None, anns_obj_id=None):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if visualize:
            cv2.imshow("mask image", mask_image)
            cv2.waitKey(0)
        if filename is not None:
            ic(os.path.join(self.image_dir, filename).replace(".", f"_{anns_obj_id}."))
            cv2.imwrite(os.path.join(self.image_dir, filename).replace("renamed_images", "renamed_images_anns").replace(".", f"_{anns_obj_id}."), mask_image*255)

        self.set_main_image(mask_image)

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
        points = generate_uniform_grid(height=height, width=width, num_points=num_points)
        labels = np.array([1], np.int32)
        for idx, point in enumerate(points):
            if image is not None:
                image = draw_points_on_image(image, points)
                self.set_main_image(image/255)

            _, out_obj_ids, out_mask_logits = self.predictor_video.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id+idx,
                points=[point],
                labels=labels,
            )

        # print(out_mask_logits)
        # mask_image = self.mask_to_numpy((out_mask_logits[0] > 0.0).cpu().numpy())

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

    def segment_video_callback(self, sender: str, app_data: str) -> None:
        """Segment for Selected video

        Args:
            sender (str): The ID of the widget that triggered this callback.
            app_data (str): Additional data from the widget.
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.inference_state = self.predictor_video.init_state(self.image_dir)
            self.frame_names = [
                p for p in os.listdir(self.image_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            # frame_names_copy = copy.deepcopy(self.frame_names)
            image = cv2.imread(os.path.join(self.image_dir, self.frame_names[0]))
            h, w, ch = image.shape

            ann_obj_id = -1
            ann_frame_idx = 0
            video_segments = {}  # video_segments contains the per-frame segmentation results
            labels = np.array([1], np.int32)

            for ann_frame_idx, filename in enumerate(self.frame_names):
                if video_segments == {}:
                    # Inital Frame
                    out_obj_ids, out_mask_logits = self.mask_point_generation(ann_frame_idx, ann_obj_id, height=h, width=w, num_points=16*16)
                    ann_obj_id = out_obj_ids[-1]

                else:
                    combined_mask = create_combined_mask(video_segments[ann_frame_idx])
                    enclosed_contours = find_enclosed_regions(combined_mask)
                    points = get_centroids(enclosed_contours)
                    ic(points)

                    for idx, point in enumerate(points):
                        _, out_obj_ids, out_mask_logits = self.predictor_video.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id+idx,
                            points=[point],
                            labels=labels,
                        )

                # Tracking
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor_video.propagate_in_video(self.inference_state):
                    if out_frame_idx >= ann_frame_idx:  # 以前までに検出した画像からの削除は行わない．
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                        }

                for out_frame_idx, (_, segments) in enumerate(video_segments.items()):
                    if out_frame_idx >= ann_frame_idx:  # 以前までに検出した画像からの削除は行わない．
                        # del overlap masks
                        _, overlaped_indices = filter_overlapping_masks(list(segments.values()), list(segments.keys()))

                        for overlaped_index in overlaped_indices:
                            del segments[overlaped_index]

                    if ann_frame_idx == len(self.frame_names)-1:
                        output_filename = self.frame_names[out_frame_idx]
                        for obj_idx, segment in segments.items():
                            mask = self.mask_to_numpy(segment, filename=output_filename, anns_obj_id=obj_idx)

                # self.del_inference_state_image()
                # input(f"Complete: {self.frame_names[ann_frame_idx]}  Next >>>")

    def _setup_gui(self) -> None:
        """Setup the Dear PyGUI layout."""
        dpg.create_context()
        dpg.create_viewport(title='Tracking Anything Editor', width=1550, height=1200)

        self.blank_main_image = np.full((self.main_image_height, self.main_image_width, 4), 100, dtype=np.uint8)
        self.blank_sub_image = np.full((self.sub_image_height, self.sub_image_width, 4), 30, dtype=np.uint8)

        with dpg.texture_registry():
            # self.temp = dpg.add_dynamic_texture(width=self.image_show_size, height=self.image_show_size, default_value=blank_image, tag="image_texture")
            self.select_image_texture = dpg.add_dynamic_texture(width=self.main_image_width, height=self.main_image_height, default_value=self.blank_main_image, tag="selected_image_texture")
            self.before_image_texture = dpg.add_dynamic_texture(width=self.sub_image_width, height=self.sub_image_height, default_value=self.blank_sub_image, tag="before_image_texture")
            self.next_image_texture = dpg.add_dynamic_texture(width=self.sub_image_width, height=self.sub_image_height, default_value=self.blank_sub_image,  tag="next_image_texture")

        with dpg.window(label="Image Processing GUI"):
            # self.text_current_frame_widget_id = dpg.add_text("frame_name", tag="frame_name")
            with dpg.group(horizontal=True):
                with dpg.group(horizontal=False):
                    dpg.add_image("before_image_texture", label="before image", tag="before_image_widget")
                    dpg.add_image("next_image_texture", label="next image", tag="next_image_widget")
                dpg.add_image("selected_image_texture", label="selected image", tag="select_image_widget")

            dpg.add_slider_int(label="Image ID", min_value=0, max_value=self.num_images-1, default_value=0, callback=self.update_images)
            dpg.add_input_text(label="Video path", default_value="/mnt/home/yuga-y/usr/splat_ws/third_party/SegAnyGAussians/models/vae_v6.pt", tag="checkpoint_path")

            dpg.add_button(label="Create Video", callback=self.create_video_callback)
            dpg.add_button(label="Segment Video", callback=self.segment_video_callback)
            dpg.add_button(label="Renane folder", callback=self.rename_folder_callback)

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
    args = parser.parse_args()

    viewer = TrackingViewer(args.input_dir, args.model, args.config)
