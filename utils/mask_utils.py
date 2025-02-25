import cv2
import torch
import numpy as np


def make_anns_image(anns, borders=True, specific_id=None):
    np.random.random(3)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for idx, ann in enumerate(sorted_anns):
        if specific_id is not None:
            if idx != specific_id:
                continue

        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    return img


def apply_mask_to_image(image, mask, convert_to_rgb=False, is_crop=False, verification_mask=False):
    """
    画像のmaskのTrueの部分のみを残し、それ以外を黒で塗りつぶす。

    Args:
        image (numpy.ndarray): 入力画像 (H, W, C) (BGR形式)
        mask (numpy.ndarray): マスク画像 (H, W) (True/False の2値)

    Returns:
        numpy.ndarray: マスクを適用した画像
    """
    # マスクを 0/1 の uint8 型に変換
    # mask = mask.squeeze()
    mask_uint8 = mask.astype(np.uint8) * 255

    # 3チャンネルのマスクを作成
    mask_3ch = cv2.merge([mask_uint8] * 3)

    # 画像とマスクを適用（False の部分を黒に）
    masked_image = cv2.bitwise_and(image, mask_3ch)

    if convert_to_rgb:
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    if is_crop:
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("cannot find contours")
            return masked_image

        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # 3. BBOX 領域のみをクロップ
        cropped_masked_image = masked_image[y:y+h, x:x+w]

        # Check if more than 80% of the cropped region is False
        if verification_mask:
            cropped_mask = mask_uint8[y:y+h, x:x+w]
            false_count = np.sum(cropped_mask == 0)
            total_count = cropped_mask.size
            if float(false_count) / float(total_count) > 0.8:
                print("============")
                print("false_count", false_count)
                print("total_count", total_count)
                print(float(false_count) / float(total_count))
                return False

        padding = 20  # パディングのサイズ (必要に応じて変更)
        padding_cropped_masked_image = cv2.copyMakeBorder(
            cropped_masked_image,
            top=padding, bottom=padding, left=padding, right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # 黒い画素
        )

        return padding_cropped_masked_image

    return masked_image


def apply_mask_to_rgba_image(image, mask):
    """
    Retain only the True parts of the mask in the RGBA image, making everything else completely black (transparent).

    Args:
        image (numpy.ndarray): Input image (H, W, 4) (RGBA format)
        mask (numpy.ndarray): Mask image, a 4-dimensional boolean array with shape (1, 1, H, W)

    Returns:
        numpy.ndarray: RGBA image with the mask applied (H, W, 4)
    """
    # Convert the mask to 2D (H, W)
    mask_2d = mask.squeeze()  # Convert to shape (H, W)

    # Create an output image initialized to black RGBA
    masked_image = np.zeros_like(image, dtype=np.uint8)

    # Apply the original image only to the regions where the mask is True
    masked_image[mask_2d] = image[mask_2d]

    return masked_image


def calculate_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.

    Args:
        mask1 (torch.Tensor): The first binary mask.
        mask2 (torch.Tensor): The second binary mask.

    Returns:
        float: The IoU of the two binary masks. Returns 0 if the union is zero.
    """
    """2つのバイナリマスクのIoUを計算"""
    intersection = torch.logical_and(mask1, mask2).sum().item()
    union = torch.logical_or(mask1, mask2).sum().item()
    return intersection / union if union > 0 else 0


def filter_overlapping_masks(masks, keys, iou_threshold=0.8, del_method="smaller"):
    """
    マスクのリストから、IoUが閾値以上のペアのうち、小さい方、または番号が後のものを削除する。
    :param masks: list of torch.Tensor (shape: [1, H, W], dtype=bool)
    :param keys: list, マスクに対応する識別キー
    :param iou_threshold: float, IoUの閾値
    :param del_method: str, "smaller"（面積の小さいほう）または "later"（番号が後のもの）を削除
    :return: tuple (filtered_masks, removed_indices)
    """
    keep = [True] * len(masks)  # マスクを保持するかどうかのフラグ
    removed_indices = []  # 削除されたマスクのインデックスリスト

    for i in range(len(masks)):
        if not keep[i]:  # すでに削除予定のものはスキップ
            continue
        for j in range(i + 1, len(masks)):
            if not keep[j]:  # すでに削除予定のものはスキップ
                continue

            iou = calculate_iou(torch.from_numpy(masks[i]), torch.from_numpy(masks[j]))

            if iou >= iou_threshold:
                if del_method == "smaller":
                    # 面積を比較し、小さい方を削除
                    area_i = torch.from_numpy(masks[i]).sum().item()
                    area_j = torch.from_numpy(masks[j]).sum().item()

                    if area_i < area_j:
                        keep[i] = False
                        removed_indices.append(keys[i])
                        break  # i が削除された場合、他との比較は不要
                    else:
                        keep[j] = False
                        removed_indices.append(keys[j])

                elif del_method == "later":
                    # 番号が後のもの（インデックス j のほう）を削除
                    keep[j] = False
                    removed_indices.append(keys[j])

    filtered_masks = [masks[i] for i in range(len(masks)) if keep[i]]
    return filtered_masks, removed_indices


def has_same_mask(target_mask: np.array, masks: dict, th: float = 0.5) -> int:
    for key, mask in masks.items():
        iou = calculate_iou(torch.from_numpy(target_mask), torch.from_numpy(mask).squeeze())
        # 重なりの大きいマスクが見つかった場合にTrue
        if iou > th:
            return key

    return -1


def calculate_overlap_ratio(combined_mask, target_mask):
    """
    Calculate the overlap ratio between two binary masks.

    The overlap ratio is defined as the intersection area of the combined_mask
    and target_mask divided by the area of the target_mask.

    Parameters:
    combined_mask (numpy.ndarray): A binary mask where the overlapping regions are to be calculated.
    target_mask (numpy.ndarray): A binary mask representing the target area.

    Returns:
    float: The ratio of the intersection area to the target area. Returns 0 if the target area is zero.
    """
    intersection = np.logical_and(combined_mask, target_mask).sum()
    target_area = target_mask.sum()
    if target_area == 0:
        return 0
    return intersection / target_area


def create_combined_mask(video_segments):
    """
    複数のマスクを OR 演算で統合し、全体のマスクを作成する
    :param video_segments: dict, マスクの辞書（キーはインデックス、値はバイナリマスク）
    :return: np.ndarray, 統合マスク (dtype=np.uint8, 0 or 255)
    """
    masks = list(video_segments.values())  # マスクのリスト取得
    combined_mask = np.logical_or.reduce(masks)  # 全てのマスクを OR で統合
    combined_mask = (combined_mask * 255).astype(np.uint8)  # 0 (False) / 255 (True) に変換
    return combined_mask[0]  # shape=(1, H, W) なので、最初の次元を削除


def find_enclosed_regions(mask):
    """
    0で囲まれている領域（穴）を検出する
    :param mask: np.ndarray, バイナリマスク (0 or 255)
    :return: List of contours
    """
    # 反転マスクを作成 (0を255, 255を0に)
    inverted_mask = cv2.bitwise_not(mask)

    # 輪郭検出 (外部輪郭のみ)
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 内部の穴の輪郭のみを抽出
    enclosed_contours = []
    for contour in contours:
        if cv2.pointPolygonTest(contour, (0, 0), False) < 0:  # 画像の外周と繋がっていない領域を取得
            enclosed_contours.append(contour)

    return enclosed_contours


def get_centroids(contours, min_area=50):
    """
    領域の大きさを検証し、一定以下のものを省いた上で、重心点のリストを返す

    :param contours: List of contours (各領域の輪郭)
    :param min_area: int, 検出対象とする最小面積
    :return: List of (x, y) tuples (重心点のリスト)
    """
    centroids = []

    for contour in contours:
        area = cv2.contourArea(contour)  # 面積を計算
        if area < min_area:
            continue  # 面積が小さいものはスキップ

        # 重心の計算 (モーメントを使用)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # X座標の重心
            cy = int(M["m01"] / M["m00"])  # Y座標の重心
            centroids.append((cx, cy))

    return centroids


def combined_all_mask(video_segment_frame):
    target_mask = video_segment_frame[1].squeeze()
    for key, mask in video_segment_frame.items():
        # すべてのマスクを統合（どれか1つでも True なら True）
        target_mask = np.logical_or.reduce([target_mask, mask.squeeze()])

    return target_mask


def visualize_instance(masks: dict) -> np.ndarray:
    """
    Visualize instance masks by assigning random colors to each unique label.

    Args:
        masks (dict): A dictionary where keys are labels and values are binary masks (numpy.ndarray).

    Returns:
        numpy.ndarray: An RGB image with the instance masks visualized.
    """
    height, width = next(iter(masks.values())).shape[1:]
    output_array = np.full((height, width), -1, dtype=int)

    # Save masks in lists
    for label, mask in masks.items():
        output_array[mask[0]] = label

    # Create an output image with 3 channels (RGB)
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Fix the random seed for reproducibility
    np.random.seed(42)

    # Assign random colors to each label
    label_colors = {label: np.random.randint(0, 255, size=3) for label in masks.keys()}

    # Color the output image based on the labels
    for label, color in label_colors.items():
        output_image[output_array == label] = color

    # Convert to RGBA
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGBA)

    # Draw the label number at the center of each region
    for label, mask in masks.items():
        # Find contours of the mask
        contours, _ = cv2.findContours(mask[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Draw the label number at the centroid
                cv2.putText(output_image, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1, cv2.LINE_AA)

    return output_image
