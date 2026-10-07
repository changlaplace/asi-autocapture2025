"""Capture one letter at calibrated settings and plot actual vs predicted RAW16."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import screeninfo

from asi_camera import ASICamera, asi
from display import Display
from forward_model import ForwardModel, read_raw16, render_display_canvas
from capture_imgnet import load_display_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', default='assets_letters/01_A.png')
    parser.add_argument('--model', default='forward_model.npz')
    parser.add_argument('--output', default='forward_model_A_comparison.png')
    parser.add_argument('--measurement', help='Use a saved RAW16 .npy or image instead of capturing again')
    parser.add_argument('--screen', type=int, default=1)
    parser.add_argument('--scale', type=float, default=0.9)
    parser.add_argument('--settle-time', type=float, default=0.7)
    args = parser.parse_args()

    model = ForwardModel.load(args.model)
    monitor_list = screeninfo.get_monitors()
    if not 0 <= args.screen < len(monitor_list):
        raise ValueError(f'Invalid screen {args.screen}; found {len(monitor_list)} monitor(s)')
    monitor = monitor_list[args.screen]
    if (monitor.height, monitor.width) != model.display_shape:
        raise ValueError(
            f'Model expects screen {model.display_shape[::-1]}, but selected screen is '
            f'{monitor.width}x{monitor.height}'
        )

    with Image.open(args.asset) as im:
        source_rgb = np.asarray(im.convert('RGB'), dtype=np.uint8)
    letter_label = Path(args.asset).stem.split('_')[-1]
    display_bgr = load_display_image(args.asset, single_channel=True, channel='green')
    canvas = render_display_canvas(
        source_rgb, model.display_shape, scale=args.scale, channel='green'
    )
    predicted = model.predict(canvas)

    if args.measurement:
        if Path(args.measurement).suffix.lower() == '.npy':
            actual = np.load(args.measurement)
        else:
            actual = read_raw16(args.measurement)
    else:
        display = Display(screen_id=args.screen)
        camera = ASICamera(camera_id=0)
        try:
            display.start_display(display_bgr, scale=args.scale, full_screen=True)
            deadline = time.monotonic() + 10
            while not display.display_flag.value:
                if display.display_proc is not None and not display.display_proc.is_alive():
                    raise RuntimeError('Display process exited before letter A appeared')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Timed out waiting for letter A to appear')
                time.sleep(0.05)
            time.sleep(args.settle_time)
            actual = camera.capture_image(
                exposure=model.exposure_us / 1e6,
                gain=model.gain,
                set_image_type=asi.ASI_IMG_RAW16,
            )
            if actual.ndim != 2 or actual.dtype != np.uint16:
                raise RuntimeError(f'Expected monochrome RAW16, received {actual.dtype} {actual.shape}')
        finally:
            display.close()
            camera.close()

    if actual.shape != predicted.shape:
        raise ValueError(f'Camera frame {actual.shape} does not match model {predicted.shape}')

    actual_f = actual.astype(np.float32)
    residual = predicted - actual_f
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    signal = float(np.mean(actual_f - model.dark))
    relative = rmse / max(signal, 1.0)
    correlation = float(np.corrcoef(
        (actual_f - model.dark).ravel(), (predicted - model.dark).ravel()
    )[0, 1])

    actual_path = Path(f'forward_model_{letter_label}_actual_raw16.npy')
    predicted_path = Path(f'forward_model_{letter_label}_predicted_raw16.npy')
    np.save(actual_path, actual)
    np.save(predicted_path, predicted)
    common_vmax = float(np.percentile(np.concatenate((actual_f.ravel(), predicted.ravel())), 99.8))
    error_scale = max(float(np.percentile(np.abs(residual), 99.5)), 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    im0 = axes[0].imshow(actual_f, cmap='gray', vmin=0, vmax=common_vmax)
    axes[0].set_title('Measured RAW16')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label='DN')
    im1 = axes[1].imshow(predicted, cmap='gray', vmin=0, vmax=common_vmax)
    axes[1].set_title('Forward-model prediction')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label='DN')
    im2 = axes[2].imshow(residual, cmap='coolwarm', vmin=-error_scale, vmax=error_scale)
    axes[2].set_title('Prediction − measurement')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label='DN')
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(
        f"Letter {letter_label} | exp={model.exposure_us / 1000:.0f} ms, gain={model.gain} | "
        f"RMSE={rmse:.1f} DN ({relative:.1%} of mean signal), "
        f"MAE={mae:.1f} DN, r={correlation:.4f}"
    )
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f'Saved comparison image: {Path(args.output).resolve()}')
    print(f'Saved actual and predicted arrays: {actual_path}, {predicted_path}')
    print(f'RMSE={rmse:.2f} DN; MAE={mae:.2f} DN; relative RMSE={relative:.4f}; r={correlation:.5f}')


if __name__ == '__main__':
    main()
