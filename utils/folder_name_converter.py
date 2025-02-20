import os
import argparse
from PIL import Image


def get_sorted_files(folder, extension):
    return sorted([f for f in os.listdir(folder) if f.endswith(extension)])


def convert_and_save_images(src_folder, extension, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = get_sorted_files(src_folder, extension)
    for file in files:
        img = Image.open(os.path.join(src_folder, file))
        base_name = os.path.splitext(file)[0].replace("frame_", "")
        new_file = os.path.join(dest_folder, f"{base_name}.jpg")
        img.convert("RGB").save(new_file, "JPEG")
        print(f"Saved {new_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and save images to a new folder.")
    parser.add_argument("--folder", type=str, default="/mnt/home/yuga-y/usr/splat_ws/datasets/shapenets/ShapeSplat2_cans_v2/images", help="Source folder containing the images.")
    parser.add_argument("--extension", type=str, default=".png", help="Extension of the files to be converted.")
    args = parser.parse_args()

    src_folder = args.folder
    extension = args.extension
    dest_folder = os.path.join(os.path.dirname(src_folder), "renamed_images")

    convert_and_save_images(src_folder, extension, dest_folder)
