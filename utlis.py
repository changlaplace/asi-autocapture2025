import os
import logging
import sys
from datetime import datetime
import re


def setup_logger(log_path):
    """
    Set up a logger that writes to a timestamped file and the terminal.

    Args:
        log_path (str): Path to the log file without extension.
    """
    pid = os.getpid()
    os.makedirs(log_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{timestamp}_{pid}.txt"

    log_dir = os.path.join(log_path, log_file)

    logger = logging.getLogger(f"{pid}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - PID %(process)d - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S',
            )
        )
        logger.addHandler(console_handler)

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
    if max_value is None or max_value < 0:
        max_value = 0
    return max_value



import os
from PIL import Image
from multiprocessing import Process, cpu_count
from tqdm import tqdm
import math

def convert_range(file_list, input_dir, output_dir):
    for filename in tqdm(file_list, desc=f"[PID {os.getpid()}]"):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_name = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(output_path):
            continue  # Already converted

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                img.save(output_path, format="PNG")
        except Exception as e:
            print(f"[{filename}] Error: {e}")

def convert_tiff_to_png(input_dir, output_dir, num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]

    if not all_files:
        print("No TIFF files found.")
        return

    num_workers = num_workers or cpu_count()
    chunk_size = math.ceil(len(all_files) / num_workers)

    workers = []
    for i in range(num_workers):
        chunk = all_files[i * chunk_size: (i + 1) * chunk_size]
        if not chunk:
            continue
        p = Process(target=convert_range, args=(chunk, input_dir, output_dir))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

def save_image(img, filename):
    """Save the captured image to a file. Used for Multprocess saving"""
    if filename is not None:
        mode = None
        if len(img.shape) == 3:
            img = img[:, :, ::-1]  # Convert BGR to RGB
        if img.dtype.name == 'uint16':
            mode = 'I;16'
        image = Image.fromarray(img, mode=mode)
        image.save(filename)

    return 

if __name__ == "__main__":
    # image_folder = r'D:\yuboz4\Imagenet_data\Hyperbolid\test'
    # current_number = get_current_captured_number(image_folder)
    # print(f"Current number of captured images in '{image_folder}': {current_number}")
    tiff_files_dir = r"D:\yuboz4\Imagenet_data\Hyperbolid\train"
    to_png_files_dir = r"C:\yuboz4\Imagenet_data\Hyperbolid\train"
    convert_tiff_to_png(tiff_files_dir, to_png_files_dir)
