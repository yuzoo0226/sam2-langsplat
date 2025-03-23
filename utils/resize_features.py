import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

WARNED = False


def display_npy_shapes(args):
    directory = args.language_features_dir

    for filename in tqdm(os.listdir(directory)):
        if args.dump_dir is None:
            args.dump_dir = os.path.join(directory, "resize_features/")
            os.makedirs(args.dump_dir, exist_ok=True)

        if filename.endswith('_s.npy'):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)
            channels, height, width = data.shape
            tqdm.write(f"{filename}: {data.shape}")

            if args.resolution == -1:
                # TODO 变更条件
                if height > args.limit_height:
                    global WARNED
                    if not WARNED:
                        print(f"[ INFO ] Encountered quite large input images (>{args.limit_height}P), rescaling to {args.limit_height}P.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = height / args.limit_height
                else:
                    global_down = 1
            else:
                global_down = width / args.resolution

            scale = float(global_down) * float(1)
            resolution = (int(width / scale), int(height / scale))
            cv_mask = data.transpose(1, 2, 0)  # Transpose to (height, width, channels)
            cv_mask_resize = cv2.resize(cv_mask, resolution)
            numpy_mask_resize = cv_mask_resize.transpose(2, 0, 1)  # Transpose back to (channels, height, width)

            tqdm.write(f"dump at: {filename} -> {os.path.join(args.dump_dir, filename)}")
            cv2.imwrite(os.path.join(args.dump_dir, f'mask_{filename.replace(".npy", args.extension)}'), cv_mask_resize*20)
            np.save(os.path.join(args.dump_dir, filename), numpy_mask_resize)
            file_path_f = file_path.replace("_s.npy", "_f.npy")
            if os.path.exists(file_path_f):
                os.makedirs(args.dump_dir, exist_ok=True)
                os.system(f"cp {file_path_f} {args.dump_dir}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize features based on resolution")
    parser.add_argument("--language_features_dir", "-i", type=str, default="/mnt/home/yuga-y/usr/splat_ws/datasets/3D_Open-vocabulary_Segmentation_datasets/room/bk_language_features/language_features")
    parser.add_argument("--dump_dir", "-d", type=str, default=None)
    parser.add_argument("--extension", type=str, default=".png")
    parser.add_argument("--resolution", "-r", type=int, default=-1, help="Resolution to resize to. Default is -1.")
    parser.add_argument("--limit_height", "-l", type=int, default=1080, help="Resolution to resize to. Default is -1.")
    args = parser.parse_args()

    display_npy_shapes(args)
