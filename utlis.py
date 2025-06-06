import os
import logging
from datetime import datetime
import re


def setup_logger(log_path):
    '''
    Setup a logger that writes to a file with the current timestamp.
    Args:
        log_path (str): Path to the log file without extension.
    '''
    pid = os.getpid()
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{timestamp}_{pid}.txt"

    log_dir = os.path.join(log_path, log_file)

    logger = logging.getLogger(f"{pid}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_dir)
        formatter = logging.Formatter('%(asctime)s - PID %(process)d - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_current_captured_number(image_folder):
    """
    Get the number of images already captured in the specified folder.
    
    Args:
        image_folder (str): Path to the folder containing captured images.
        
    Returns:
        int: The number of images in the folder.
    """
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' does not exist.")
        return 0
    max_value = -1

    for filename in os.listdir(image_folder):
        parts = filename.split('_')
        if len(parts) >= 3:

            value = int(parts[1])  
            max_value = max(max_value, value)
    if max_value < 0:
        max_value = 0
    return max_value


from PIL import Image

def convert_tiff_to_png(input_dir, output_dir):
    """
    Converts all .tiff or .tif images in input_dir to .png and saves them in output_dir.
    
    Args:
        input_dir (str): Path to the folder containing .tiff images.
        output_dir (str): Path to the folder where .png images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_name)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")  # remove alpha if needed
                    img.save(output_path, format="PNG")
                    print(f"Converted: {filename} → {output_name}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


if __name__ == "__main__":
    # image_folder = r'D:\yuboz4\Imagenet_data\Hyperbolid\test'
    # current_number = get_current_captured_number(image_folder)
    # print(f"Current number of captured images in '{image_folder}': {current_number}")
    tiff_files_dir = r"D:\yuboz4\Imagenet_data\Hyperbolid\train"
    to_png_files_dir = r"C:\yuboz4\Imagenet_data\Hyperbolid\train"
    convert_tiff_to_png(tiff_files_dir, to_png_files_dir)