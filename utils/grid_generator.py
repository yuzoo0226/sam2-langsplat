import cv2
import numpy as np
from icecream import ic
from typing import Any, Dict, List, Optional, Tuple

from sam2.modeling.sam2_base import SAM2Base
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from segment_anything_langsplat import SamAutomaticMaskGenerator


def generate_uniform_grid(height, width, num_points, padding=20):
    effective_width = width - 2 * padding
    effective_height = height - 2 * padding

    num_x = int(np.sqrt(num_points))
    num_y = num_x

    x = np.linspace(padding, padding + effective_width - 1, num_x)
    y = np.linspace(padding, padding + effective_height - 1, num_y)
    xx, yy = np.meshgrid(x, y)

    points = np.vstack([xx.ravel(), yy.ravel()]).T
    return points


def draw_points_on_image(image, points, color=(0, 0, 255), radius=3):
    output_image = image.copy()
    for (x, y) in points.astype(int):
        cv2.circle(output_image, (x, y), radius, color, -1)
    # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)
    return output_image


def auto_mask_generator(
    image: np.ndarray,
    model: SAM2Base,
    points_per_side: Optional[int] = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.8,
    stability_score_thresh: float = 0.95,
    stability_score_offset: float = 1.0,
    mask_threshold: float = 0.0,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    point_grids: Optional[List[np.ndarray]] = None,
    min_mask_region_area: int = 0,
    output_mode: str = "binary_mask",
    use_m2m: bool = False,
    multimask_output: bool = True,
    base: str = "sam2",
    mask_level: str = "large"
):
    if base == "sam":
        mask_generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            # mask_threshold=mask_threshold,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            # box_nms_thresh=box_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            point_grids=point_grids,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
            # use_m2m=use_m2m,
            # multimask_output=multimask_output
        )
        _, masks_s, masks_m, masks_l = mask_generator.generate(image)
        masks = masks_l

    else:  # if base == "sam2"
        mask_generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            mask_threshold=mask_threshold,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            box_nms_thresh=box_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            point_grids=point_grids,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
            use_m2m=use_m2m,
            multimask_output=multimask_output
        )
        masks = mask_generator.generate(image)

    for idx, mask in enumerate(masks):
        ic(idx, mask["area"])
    point_coords_array = np.array([mask['point_coords'][0] for mask in masks])
    segmentation_array = np.array([mask['segmentation'] for mask in masks])

    return point_coords_array, segmentation_array, masks


if __name__ == "__main__":
    height, width = 480, 640
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    num_points = 32
    points = generate_uniform_grid(height, width, num_points)

    print(points)

    output_image = draw_points_on_image(image, points)

    cv2.imshow("Points on Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
