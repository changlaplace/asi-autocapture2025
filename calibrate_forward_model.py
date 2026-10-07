"""Acquire a fixed-setting RAW16 calibration set for ``forward_model.py``.

Run this on the optical setup with the intended display, channel, screen and
camera settings. The displayed calibration frame always fills the selected
monitor, so use the same screen geometry for later predictions.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import screeninfo

from asi_camera import ASICamera, asi
from display import Display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='forward_calibration')
    parser.add_argument('--screen', type=int, default=1)
    parser.add_argument('--exposure', type=float, default=0.01, help='Fixed exposure in seconds')
    parser.add_argument('--gain', type=int, default=0)
    parser.add_argument('--settle-time', type=float, default=0.5)
    parser.add_argument('--channel', choices=['red', 'green', 'blue'], default='green')
    args = parser.parse_args()

    monitors = screeninfo.get_monitors()
    if not 0 <= args.screen < len(monitors):
        raise ValueError(f'Invalid screen {args.screen}; found {len(monitors)} monitor(s)')
    monitor = monitors[args.screen]
    height, width = monitor.height, monitor.width
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    channel = {'red': 2, 'green': 1, 'blue': 0}[args.channel]  # Display expects BGR
    display = Display(screen_id=args.screen)
    camera = ASICamera(camera_id=0)

    def capture(name, level=None, impulse=False, pattern=None):
        frame = np.zeros((height, width, 3), np.uint8)
        if impulse:
            # Small square provides a compact PSF probe while surviving display scaling.
            side = max(8, min(width, height) // 48)
            x0, y0 = (width - side) // 2, (height - side) // 2
            frame[y0:y0 + side, x0:x0 + side, channel] = 255
        elif level is not None:
            frame[:, :, channel] = level
        elif pattern == 'half':
            frame[:, :width // 2, channel] = 255
        elif pattern == 'stripes':
            frame[:, ::2, channel] = 255
        elif pattern == 'checker':
            yy, xx = np.indices((height, width))
            frame[((yy // 48 + xx // 48) % 2) == 0, channel] = 255
        display.start_display(frame, scale=1.0, full_screen=True)
        deadline = time.monotonic() + 10
        while not display.display_flag.value:
            if display.display_proc is not None and not display.display_proc.is_alive():
                raise RuntimeError('Display process exited before showing calibration pattern')
            if time.monotonic() > deadline:
                raise TimeoutError('Display did not become ready')
            time.sleep(0.05)
        time.sleep(args.settle_time)
        raw = camera.capture_image(args.exposure, args.gain, asi.ASI_IMG_RAW16)
        if raw.ndim != 2 or raw.dtype != np.uint16:
            raise RuntimeError(f'Expected monochrome RAW16, received {raw.dtype} {raw.shape}')
        np.save(out / f'{name}.npy', raw)
        print(f'{name}: shape={raw.shape}, range={int(raw.min())}..{int(raw.max())}')
        display.close()

    try:
        capture('dark', level=0)
        for level in (16, 32, 48, 64, 96, 128, 160, 192, 224, 255):
            capture(f'flat_{level:03d}', level=level)
        capture('impulse', impulse=True)
        # Held-out patterns are deliberately not used by fit_calibration.
        capture('validation_half', pattern='half')
        capture('validation_stripes', pattern='stripes')
        capture('validation_checker', pattern='checker')
        metadata = {
            'camera': camera.camera_info['Name'], 'screen': args.screen,
            'screen_width': width, 'screen_height': height,
            'exposure_us': round(args.exposure * 1e6), 'gain': args.gain,
            'channel': args.channel, 'impulse_side_px': max(8, min(width, height) // 48),
        }
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    finally:
        display.close()
        camera.close()


if __name__ == '__main__':
    main()
