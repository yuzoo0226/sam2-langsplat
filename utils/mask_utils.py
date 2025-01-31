import cv2
import torch
import numpy as np


def calculate_iou(mask1, mask2):
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