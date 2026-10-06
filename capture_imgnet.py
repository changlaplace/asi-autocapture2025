import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from asi_camera import ASICamera, asi
from display import Display
from utlis import setup_logger


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def get_image_files(asset_dir):
    """Return supported image files in stable filename order."""
    files = sorted(
        path for path in Path(asset_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f'No images found in {Path(asset_dir).resolve()}')
    return files


def load_display_image(path, single_channel=False, channel='green'):
    """Load an image as BGR uint8 data for OpenCV display."""
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    if single_channel:
        channel_map = {'r': 0, 'red': 0, 'g': 1, 'green': 1, 'b': 2, 'blue': 2}
        key = channel.lower()
        if key not in channel_map:
            raise ValueError(f'Unsupported display channel: {channel}')
        display_rgb = np.zeros_like(rgb)
        display_rgb[:, :, channel_map[key]] = rgb[:, :, channel_map[key]]
        rgb = display_rgb
    return np.ascontiguousarray(rgb[:, :, ::-1])


def wait_for_display(display, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not display.display_flag.value:
        if display.display_proc is not None and not display.display_proc.is_alive():
            raise RuntimeError('The display process exited before showing the image.')
        if time.monotonic() >= deadline:
            raise TimeoutError('Timed out while waiting for the display window.')
        time.sleep(0.05)


def capture_asset_images(
    asset_dir,
    output_dir,
    exposure=0.001,
    gain=0,
    scale=0.9,
    screen=1,
    settle_time=0.5,
    start_index=0,
    limit=None,
    single_channel_display=True,
    display_channel='green',
):
    image_files = get_image_files(asset_dir)[start_index:]
    if limit is not None:
        image_files = image_files[:limit]
    if not image_files:
        raise ValueError('No images remain after applying --start-index and --limit.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('capturing_logs')
    camera = ASICamera(camera_id=0)
    display = Display(screen_id=screen)
    current_exposure = exposure

    logger.info(
        'Starting capture: %d images, camera=%s, screen=%d',
        len(image_files),
        camera.camera_info['Name'],
        screen,
    )

    try:
        for position, image_path in enumerate(image_files, start=1):
            source_index = start_index + position
            logger.info(
                'Displaying %s (%d/%d)',
                image_path.name,
                position,
                len(image_files),
            )
            display.start_display(
                load_display_image(
                    image_path,
                    single_channel=single_channel_display,
                    channel=display_channel,
                ),
                scale=scale,
                full_screen=True,
            )
            wait_for_display(display)
            time.sleep(settle_time)

            for attempt in range(12):
                captured = camera.capture_image(
                    exposure=current_exposure,
                    gain=gain,
                    set_image_type=asi.ASI_IMG_RAW8,
                )
                high_percentile = float(np.percentile(captured, 99))
                saturated_fraction = float(np.mean(captured >= 250))
                if 180 <= high_percentile <= 245 and saturated_fraction < 0.01:
                    break

                previous_exposure = current_exposure
                if high_percentile > 245 or saturated_fraction >= 0.01:
                    current_exposure = max(current_exposure * 0.5, 32e-6)
                else:
                    brightness_ratio = 210 / max(high_percentile, 1)
                    current_exposure = min(current_exposure * min(brightness_ratio, 2), 1.0)

                logger.info(
                    'Auto exposure %d for %s: p99=%.1f, saturated=%.2f%%, %.6fs -> %.6fs',
                    attempt + 1,
                    image_path.name,
                    high_percentile,
                    saturated_fraction * 100,
                    previous_exposure,
                    current_exposure,
                )
                if current_exposure == previous_exposure:
                    break
            else:
                logger.warning(
                    '%s did not reach the target brightness after 12 attempts; '
                    'saving the latest frame.',
                    image_path.name,
                )

            output_path = output_dir / (
                f'{source_index:03d}_{image_path.stem}_captured.png'
            )
            camera.save_captured_image(
                captured,
                output_path,
                set_image_type=asi.ASI_IMG_RAW8,
            )
            logger.info(
                'Saved %s (exposure=%.6fs, gain=%d, range=%d-%d)',
                output_path,
                current_exposure,
                gain,
                int(captured.min()),
                int(captured.max()),
            )
            print(f'[{position:02d}/{len(image_files):02d}] {output_path}')
    finally:
        display.close()
        camera.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Display local images one by one and capture them with a ZWO ASI camera.'
    )
    parser.add_argument('--assets', default='assets_letters', help='Directory containing source images.')
    parser.add_argument('--output', default='capdata_letters', help='Directory for captured images.')
    parser.add_argument('--exposure', type=float, default=1.0, help='Initial exposure in seconds.')
    parser.add_argument('--gain', type=int, default=0, help='Camera gain value.')
    parser.add_argument('--scale', type=float, default=0.9, help='Image scale relative to the selected screen.')
    parser.add_argument('--screen', type=int, default=1, help='Zero-based monitor index used for display.')
    parser.add_argument('--settle-time', type=float, default=0.5, help='Seconds to wait before each capture.')
    parser.add_argument('--start-index', type=int, default=0, help='Zero-based source image index to start at.')
    parser.add_argument('--limit', type=int, help='Maximum number of images to capture.')
    parser.add_argument(
        '--single-channel-display',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Display only one RGB channel in that channel color instead of the full RGB image.',
    )
    parser.add_argument(
        '--display-channel',
        choices=['r', 'g', 'b', 'red', 'green', 'blue'],
        default='green',
        help='Channel to isolate when --single-channel-display is enabled.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    capture_asset_images(
        asset_dir=args.assets,
        output_dir=args.output,
        exposure=args.exposure,
        gain=args.gain,
        scale=args.scale,
        screen=args.screen,
        settle_time=args.settle_time,
        start_index=args.start_index,
        limit=args.limit,
        single_channel_display=args.single_channel_display,
        display_channel=args.display_channel,
    )
