"""Capture repeated long-exposure RAW16 frames for a cleaner spot-array PSF."""

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
    parser.add_argument('--exposure', type=float, default=1.0)
    parser.add_argument('--gain', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--settle-time', type=float, default=0.5)
    args = parser.parse_args()
    if args.exposure < 0.1:
        raise ValueError('Use at least 0.1 second for the long-exposure PSF capture')
    monitors = screeninfo.get_monitors()
    if not 0 <= args.screen < len(monitors):
        raise ValueError(f'Invalid screen {args.screen}; found {len(monitors)} monitor(s)')
    monitor = monitors[args.screen]
    h, w = monitor.height, monitor.width
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    channel = 1  # BGR green; monochrome camera and calibrated green channel.
    display = Display(screen_id=args.screen)
    camera = ASICamera(camera_id=0)

    def capture_mean(name, offset=(0, 0), probe=True):
        frame = np.zeros((h, w, 3), np.uint8)
        side = max(8, min(w, h) // 48)
        x0 = (w - side) // 2 + int(offset[0])
        y0 = (h - side) // 2 + int(offset[1])
        if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
            raise ValueError('PSF probe offset moves the square outside the screen')
        if probe:
            frame[y0:y0 + side, x0:x0 + side, channel] = 255
        display.start_display(frame, scale=1.0, full_screen=True)
        deadline = time.monotonic() + 10
        while not display.display_flag.value:
            if display.display_proc is not None and not display.display_proc.is_alive():
                raise RuntimeError('Display process exited before showing the PSF pattern')
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for PSF pattern display')
            time.sleep(0.05)
        time.sleep(args.settle_time)
        frames = []
        for index in range(args.repeats):
            raw = camera.capture_image(args.exposure, args.gain, asi.ASI_IMG_RAW16)
            if raw.ndim != 2 or raw.dtype != np.uint16:
                raise RuntimeError(f'Expected monochrome RAW16, got {raw.dtype} {raw.shape}')
            frames.append(raw.astype(np.float32))
            print(f'{name} {index + 1}/{args.repeats}: {int(raw.min())}..{int(raw.max())}')
        display.close()
        mean = np.mean(frames, axis=0, dtype=np.float32)
        np.save(out / f'{name}_mean.npy', mean)
        return mean

    try:
        dark = capture_mean('psf_dark', probe=False)
        probe = capture_mean('psf_probe', offset=(0, 0))
        capture_mean('psf_probe_x', offset=(w // 4, 0))
        capture_mean('psf_probe_y', offset=(0, h // 4))
        metadata = {
            'screen': args.screen, 'screen_width': w, 'screen_height': h,
            'exposure_us': round(args.exposure * 1e6), 'gain': args.gain,
            'channel': 'green', 'probe_side_px': max(8, min(w, h) // 48),
            'repeats': args.repeats,
            'probe_offsets_px': {'center': [0, 0], 'x': [w // 4, 0], 'y': [0, h // 4]},
            'dark_max': float(dark.max()), 'probe_max': float(probe.max()),
        }
        (out / 'psf_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        delta = np.maximum(probe - dark, 0)
        print(f'Mean PSF signal: total={delta.sum():.0f}, max pixel={delta.max():.1f} DN')
        if max(float(dark.max()), float(probe.max())) >= 65000:
            print('WARNING: PSF capture is near RAW16 saturation; reduce --exposure and repeat.')
    finally:
        display.close()
        camera.close()


if __name__ == '__main__':
    main()
