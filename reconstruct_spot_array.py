"""Recover a display-plane letter by folding and averaging measured spot copies.

The full-array PSF has deep Fourier nulls, so direct Wiener inversion is
ill-conditioned. This lightweight reconstruction instead aligns the repeated
letter images using the calibrated spot centers, averages them, and uses the
calibrated magnification to map the recovered tile back to display scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forward_model import ForwardModel, render_display_canvas


def _cluster_axis(values: np.ndarray, gap: float = 30.0) -> np.ndarray:
    ordered = np.sort(values)
    groups: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if value - groups[-1][-1] > gap:
            groups.append([float(value)])
        else:
            groups[-1].append(float(value))
    return np.asarray([np.median(group) for group in groups], np.float32)


def reconstruct_spot_average(
    measurement: np.ndarray, model: ForwardModel
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a display-sized reconstruction, folded spot tile, and copy count."""
    if measurement.shape != model.dark.shape:
        raise ValueError(f"Expected measurement shape {model.dark.shape}, got {measurement.shape}")
    if model.spots_yx is None or model.magnification_xy is None:
        raise ValueError("The fitted model must include spot centers and magnification")

    # Flat-field correction is floored at the 10th percentile to avoid
    # amplifying noise from very insensitive or dead pixels.
    gain_floor = max(float(np.percentile(model.flat_gain, 10)), 1.0)
    signal = (measurement.astype(np.float32) - model.dark)
    signal /= np.maximum(model.flat_gain, gain_floor) * model.signal_scale

    ys = _cluster_axis(model.spots_yx[:, 0])
    xs = _cluster_axis(model.spots_yx[:, 1])
    if len(ys) < 2 or len(xs) < 2:
        raise ValueError("Could not estimate the two-dimensional spot lattice pitch")
    # Crop the central 80% of each lattice cell. Adjacent display replicas
    # overlap near cell boundaries; excluding that region reduces ghost edges.
    tile_h = max(8, int(round(0.8 * float(np.median(np.diff(ys))))))
    tile_w = max(8, int(round(0.8 * float(np.median(np.diff(xs))))))

    tiles = []
    for cy, cx in model.spots_yx:
        y0 = int(round(float(cy) - tile_h / 2))
        x0 = int(round(float(cx) - tile_w / 2))
        y1, x1 = y0 + tile_h, x0 + tile_w
        if y0 < 0 or x0 < 0 or y1 > signal.shape[0] or x1 > signal.shape[1]:
            continue
        tile = signal[y0:y1, x0:x1].copy()
        border = np.concatenate((tile[:4].ravel(), tile[-4:].ravel(),
                                 tile[:, :4].ravel(), tile[:, -4:].ravel()))
        tile -= float(np.median(border))
        tiles.append(tile)
    if not tiles:
        raise ValueError("No complete spot cells fit inside the measurement")

    # A median fold rejects lenslet-specific hot pixels, flare, and weak
    # misregistered copies while preserving the common displayed pattern.
    folded = np.median(np.stack(tiles), axis=0)
    # Folded-cell background varies slightly across the detector. Set a robust
    # shared floor to the middle of the folded distribution before contrast
    # mapping so that weak lenslet haze does not become a gray screen.
    lo, hi = np.percentile(folded, [60, 99.5])
    if hi <= lo:
        raise ValueError("Folded measurement has no useful image contrast")
    folded_linear = np.clip((folded - lo) / (hi - lo), 0, 1)

    # Display the recovered tile with calibrated display gamma. Contrast is
    # normalized from the measurement because per-cell flux is not an absolute
    # display-code calibration.
    display_codes = np.interp(
        folded_linear, model.response, np.arange(256, dtype=np.float32)
    ).astype(np.uint8)
    mag_x, mag_y = map(abs, model.magnification_xy)
    patch_w = max(tile_w, int(round(tile_w / max(mag_x, 1e-6))))
    patch_h = max(tile_h, int(round(tile_h / max(mag_y, 1e-6))))
    patch = cv2.resize(display_codes, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)

    screen_h, screen_w = model.display_shape
    canvas = np.zeros((screen_h, screen_w), np.uint8)
    x0, y0 = (screen_w - patch_w) // 2, (screen_h - patch_h) // 2
    ox0, oy0 = max(0, x0), max(0, y0)
    px0, py0 = max(0, -x0), max(0, -y0)
    copy_w, copy_h = min(screen_w - ox0, patch_w - px0), min(screen_h - oy0, patch_h - py0)
    canvas[oy0:oy0 + copy_h, ox0:ox0 + copy_w] = patch[
        py0:py0 + copy_h, px0:px0 + copy_w
    ]
    return canvas, display_codes, len(tiles)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", help="Saved RAW16 measurement (.npy or image)")
    parser.add_argument("--model", default="forward_model.npz")
    parser.add_argument("--reference", help="Optional displayed image for evaluation only")
    parser.add_argument("--output-dir", default="forward_model_validation/reconstructions")
    parser.add_argument("--scale", type=float, default=0.9,
                        help="Display scale used for the measurement (default: 0.9)")
    args = parser.parse_args()

    measurement_path = Path(args.measurement)
    if measurement_path.suffix.lower() == ".npy":
        measurement = np.load(measurement_path).astype(np.float32)
    else:
        with Image.open(measurement_path) as im:
            measurement = np.asarray(im).astype(np.float32)
    model = ForwardModel.load(args.model)
    reconstruction, folded, copies = reconstruct_spot_average(measurement, model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = measurement_path.stem.replace("forward_model_", "").replace("_actual_raw16", "")
    reconstruction_path = out_dir / f"{label}_spot_reconstruction.png"
    folded_path = out_dir / f"{label}_folded_spot_tile.png"
    comparison_path = out_dir / f"{label}_spot_reconstruction_comparison.png"
    Image.fromarray(reconstruction, mode="L").save(reconstruction_path)
    Image.fromarray(folded, mode="L").save(folded_path)

    panels = 3 if args.reference else 2
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = float(np.percentile(measurement, 99.8))
    axes[0].imshow(measurement, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title("Measured RAW16")
    axes[1].imshow(reconstruction, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Spot-fold reconstruction ({copies} copies averaged)")
    if args.reference:
        with Image.open(args.reference) as im:
            reference = np.asarray(im.convert("RGB"), dtype=np.uint8)
        reference = render_display_canvas(
            reference, model.display_shape, scale=args.scale, channel="green"
        )
        axes[2].imshow(reference, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Displayed reference (evaluation only)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("Measurement-only reconstruction from the calibrated spot lattice")
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)

    print(f"Averaged {copies} measured spot copies")
    print(f"Saved display-scale reconstruction: {reconstruction_path.resolve()}")
    print(f"Saved folded tile: {folded_path.resolve()}")
    print(f"Saved comparison: {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
