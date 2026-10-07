"""Compact display-to-camera model using the measured lenslet spot lattice.

It predicts a diffuse response plus scaled copies of the display at the
measured spot centers. The compact fit also stores RAW16 background, flat-field
gain, and the display's measured intensity response curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def read_raw16(path: str | Path) -> np.ndarray:
    """Read a saved 16-bit monochrome camera frame without reducing bit depth."""
    with Image.open(path) as image:
        frame = np.asarray(image)
    if frame.ndim != 2:
        raise ValueError(f"Expected a monochrome RAW16 frame, got shape {frame.shape}")
    if frame.dtype != np.uint16:
        if frame.dtype.kind not in "ui" or frame.max(initial=0) > 65535:
            raise ValueError(f"Expected 16-bit integer data, got {frame.dtype}")
        frame = frame.astype(np.uint16)
    return frame


def _convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear convolution with zero padding and output matching image shape."""
    shape = (image.shape[0] + kernel.shape[0] - 1,
             image.shape[1] + kernel.shape[1] - 1)
    full = np.fft.irfft2(
        np.fft.rfft2(image, shape) * np.fft.rfft2(kernel, shape), shape
    )
    y0, x0 = (kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2
    return full[y0:y0 + image.shape[0], x0:x0 + image.shape[1]]


@dataclass
class ForwardModel:
    """Fitted monochrome forward model with an efficient compact PSF."""

    dark: np.ndarray
    flat_gain: np.ndarray
    psf: np.ndarray
    response: np.ndarray
    exposure_us: int
    gain: int
    display_shape: tuple[int, int]
    spots_yx: np.ndarray | None = None
    spot_weights: np.ndarray | None = None
    magnification_xy: tuple[float, float] | None = None
    spot_mix: float | None = None
    psf_sampling_xy: tuple[float, float] = (1.0, 1.0)
    signal_scale: float = 1.0

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, dark=self.dark, flat_gain=self.flat_gain, psf=self.psf,
            response=self.response, exposure_us=self.exposure_us, gain=self.gain,
            display_shape=np.asarray(self.display_shape, np.int32),
            spots_yx=np.asarray(self.spots_yx if self.spots_yx is not None else [], np.float32),
            spot_weights=np.asarray(self.spot_weights if self.spot_weights is not None else [], np.float32),
            magnification_xy=np.asarray(
                self.magnification_xy if self.magnification_xy is not None else [0, 0], np.float32
            ),
            spot_mix=np.asarray(self.spot_mix if self.spot_mix is not None else 0, np.float32),
            psf_sampling_xy=np.asarray(self.psf_sampling_xy, np.float32),
            signal_scale=np.asarray(self.signal_scale, np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ForwardModel":
        with np.load(path) as d:
            return cls(
                dark=d["dark"].astype(np.float32),
                flat_gain=d["flat_gain"].astype(np.float32),
                psf=d["psf"].astype(np.float32),
                response=d["response"].astype(np.float32),
                exposure_us=int(d["exposure_us"]),
                gain=int(d["gain"]),
                display_shape=tuple(map(int, d["display_shape"])),
                spots_yx=d["spots_yx"].astype(np.float32)
                if "spots_yx" in d and d["spots_yx"].size else None,
                spot_weights=d["spot_weights"].astype(np.float32)
                if "spot_weights" in d and d["spot_weights"].size else None,
                magnification_xy=tuple(map(float, d["magnification_xy"]))
                if "magnification_xy" in d and np.any(d["magnification_xy"]) else None,
                spot_mix=float(d["spot_mix"]) if "spot_mix" in d else None,
                psf_sampling_xy=tuple(map(float, d["psf_sampling_xy"]))
                if "psf_sampling_xy" in d else (1.0, 1.0),
                signal_scale=float(d["signal_scale"]) if "signal_scale" in d else 1.0,
            )

    def predict(self, display_image: np.ndarray) -> np.ndarray:
        """Predict RAW16 values for a grayscale uint8 display image.

        Input is the final full-screen grayscale uint8 canvas (or RGB canvas)
        sent to Display, before its full-screen resize. The result is float32
        in RAW16 digital-number units.
        """
        from cv2 import INTER_AREA, resize

        src = np.asarray(display_image)
        if src.ndim == 3:
            # Monochrome camera with the project's usual isolated green display.
            src = src[..., 1]
        if src.ndim != 2:
            raise ValueError("display_image must be HxW grayscale or RGB")
        if src.dtype != np.uint8:
            raise ValueError("display_image must contain uint8 display code values")
        dh, dw = self.display_shape
        if src.shape != (dh, dw):
            raise ValueError(
                f"Expected final display canvas {(dh, dw)}, got {src.shape}; "
                "apply the same centering/scaling used for display first."
            )
        linear_screen = self.response[src]
        if self.spots_yx is not None and self.spot_weights is not None and self.magnification_xy:
            spot_optical = _render_spot_array(
                linear_screen, self.spots_yx, self.spot_weights,
                self.magnification_xy, self.dark.shape,
            )
            displayed = resize(src, (self.dark.shape[1], self.dark.shape[0]),
                               interpolation=INTER_AREA).astype(np.uint8)
            linear = self.response[displayed]
            linear = _centered_scale(linear, self.psf_sampling_xy)
            smooth = _convolve_same(linear, self.psf)
            smooth /= np.maximum(_convolve_same(np.ones_like(linear), self.psf), 1e-6)
            mix = float(np.clip(self.spot_mix if self.spot_mix is not None else 1.0, 0, 1))
            optical = smooth * (1 - mix) + spot_optical * mix
            pred = self.dark + optical * self.flat_gain * self.signal_scale
            return np.clip(pred, 0, 65535).astype(np.float32)

        displayed = resize(src, (self.dark.shape[1], self.dark.shape[0]),
                           interpolation=INTER_AREA).astype(np.uint8)
        linear = self.response[displayed]
        linear = _centered_scale(linear, self.psf_sampling_xy)
        # Flat calibration captures a uniform field. flat_gain corrects fixed
        # pixel response / illumination falloff; PSF carries spatial mixing.
        optical = _convolve_same(linear, self.psf)
        edge_norm = _convolve_same(np.ones_like(linear), self.psf)
        optical /= np.maximum(edge_norm, 1e-6)
        if optical.shape != self.dark.shape:
            from cv2 import resize as cv_resize
            optical = cv_resize(optical, (self.dark.shape[1], self.dark.shape[0]))
        pred = self.dark + optical * self.flat_gain * self.signal_scale
        return np.clip(pred, 0, 65535).astype(np.float32)


def _render_spot_array(
    display_linear: np.ndarray,
    spots_yx: np.ndarray,
    weights: np.ndarray,
    magnification_xy: tuple[float, float],
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Render a demagnified copy of the display at each measured lenslet spot."""
    import cv2

    mag_x, mag_y = magnification_xy
    out_h, out_w = output_shape
    in_h, in_w = display_linear.shape
    mini_w = max(1, int(round((in_w - 1) * abs(mag_x))) + 1)
    mini_h = max(1, int(round((in_h - 1) * abs(mag_y))) + 1)
    mini = cv2.resize(display_linear, (mini_w, mini_h), interpolation=cv2.INTER_AREA)
    # The probe-shift calibration reports spot motion, whose sign is opposite
    # the pixel orientation observed in the captured tiled letter image. Keep
    # the measured absolute sampling scale, but use the camera-observed image
    # orientation (validated with the asymmetric B pattern).
    # One small optical spot footprint, estimated by the measured point PSF.
    mini = cv2.GaussianBlur(mini, (0, 0), 1.0)
    numerator = np.zeros(output_shape, np.float32)
    denominator = np.zeros(output_shape, np.float32)
    for (cy, cx), weight in zip(spots_yx, weights):
        y0, x0 = int(round(cy - (mini_h - 1) / 2)), int(round(cx - (mini_w - 1) / 2))
        y1, x1 = y0 + mini_h, x0 + mini_w
        oy0, ox0 = max(0, y0), max(0, x0)
        oy1, ox1 = min(out_h, y1), min(out_w, x1)
        if oy0 >= oy1 or ox0 >= ox1:
            continue
        iy0, ix0 = oy0 - y0, ox0 - x0
        iy1, ix1 = iy0 + oy1 - oy0, ix0 + ox1 - ox0
        numerator[oy0:oy1, ox0:ox1] += mini[iy0:iy1, ix0:ix1] * weight
        denominator[oy0:oy1, ox0:ox1] += weight
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)


def _centered_scale(image: np.ndarray, scale_xy: tuple[float, float]) -> np.ndarray:
    """Scale a sampled camera-plane image around its center, with zero fill."""
    import cv2

    h, w = image.shape
    sx, sy = scale_xy
    new_w = max(1, int(round(w * sx)))
    new_h = max(1, int(round(h * sy)))
    interpolation = cv2.INTER_LINEAR if sx >= 1 and sy >= 1 else cv2.INTER_AREA
    scaled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    output = np.zeros_like(image, dtype=np.float32)
    x0, y0 = (w - new_w) // 2, (h - new_h) // 2
    ox0, oy0 = max(0, x0), max(0, y0)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    copy_w, copy_h = min(w - ox0, new_w - sx0), min(h - oy0, new_h - sy0)
    output[oy0:oy0 + copy_h, ox0:ox0 + copy_w] = scaled[
        sy0:sy0 + copy_h, sx0:sx0 + copy_w
    ]
    return output


def fit_calibration(directory: str | Path, output: str | Path = "forward_model.npz") -> ForwardModel:
    """Fit response curve, dark map, flat-field gain and compact PSF."""
    import json

    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    dark = np.load(root / "dark.npy").astype(np.float32)
    flat255 = np.load(root / "flat_255.npy").astype(np.float32)
    impulse = np.load(root / "impulse.npy").astype(np.float32)
    if dark.shape != flat255.shape or dark.shape != impulse.shape:
        raise ValueError("Calibration frames have inconsistent camera dimensions")

    # Scalar response curve estimated from robust full-frame averages.
    levels = np.array([16, 32, 48, 64, 96, 128, 160, 192, 224, 255], np.int32)
    means = []
    for level in levels:
        frame = np.load(root / f"flat_{level:03d}.npy").astype(np.float32)
        if frame.shape != dark.shape:
            raise ValueError(f"flat_{level:03d} has an unexpected shape")
        means.append(float(np.mean(frame - dark)))
    full_mean = float(np.mean(flat255 - dark))
    if full_mean <= 0:
        raise ValueError("White-field response is not above the measured black level")
    if float(np.percentile(flat255, 99.9)) >= 65500:
        raise ValueError("White-field RAW16 frame clips; recalibrate with shorter exposure")
    response = np.interp(np.arange(256), np.r_[0, levels],
                         np.r_[0.0, np.maximum(means, 0) / full_mean]).astype(np.float32)
    response = np.maximum.accumulate(np.clip(response, 0, 1))

    # Prefer repeated long-exposure probes. The camera shows a fixed lattice of
    # lenslet spots; display shifts move the image within each lattice cell,
    # rather than translating the full spot array. Calibrate both facts here.
    long_dark = root / "psf_dark_mean.npy"
    long_probe = root / "psf_probe_mean.npy"
    spots_yx = spot_weights = None
    magnification_xy = None
    spot_mix = None
    if long_dark.exists() and long_probe.exists():
        import cv2

        probe_raw = np.load(long_probe).astype(np.float32) - np.load(long_dark).astype(np.float32)
        if probe_raw.shape != dark.shape:
            raise ValueError("Long-exposure PSF frames have an unexpected sensor shape")
        highpass = probe_raw - cv2.GaussianBlur(probe_raw, (0, 0), 3.0)
        h, w = highpass.shape
        n = min(100, h // 4, w // 4)
        corners = np.concatenate((highpass[:n, :n].ravel(), highpass[:n, -n:].ravel(),
                                  highpass[-n:, :n].ravel(), highpass[-n:, -n:].ravel()))
        baseline = float(np.median(corners))
        noise_sigma = 1.4826 * float(np.median(np.abs(corners - baseline)))
        spot_field = np.maximum(probe_raw - baseline - 3 * noise_sigma, 0)
        spot_mix = float(spot_field.sum() / max(float(np.maximum(probe_raw, 0).sum()), 1.0))
        smoothed = cv2.GaussianBlur(spot_field, (0, 0), 1.0)
        threshold = max(8 * noise_sigma, 0.15 * float(smoothed.max()))
        maxima = cv2.dilate(smoothed, np.ones((9, 9), np.uint8))
        py, px = np.where((smoothed >= maxima) & (smoothed > threshold))
        spots_yx = np.stack((py, px), axis=1).astype(np.float32)
        if len(spots_yx) < 12:
            raise ValueError(f"Only {len(spots_yx)} PSF spots detected; increase long exposure")
        spot_weights = np.array([
            float(spot_field[max(0, int(y)-5):int(y)+6, max(0, int(x)-5):int(x)+6].sum())
            for y, x in spots_yx
        ], np.float32)
        keep = spot_weights > 0
        spots_yx, spot_weights = spots_yx[keep], spot_weights[keep]

        psf_meta_path = root / "psf_metadata.json"
        if not psf_meta_path.exists():
            raise ValueError("PSF shift captures and psf_metadata.json are required for the spot model")
        psf_meta = json.loads(psf_meta_path.read_text(encoding="utf-8"))
        offsets = psf_meta["probe_offsets_px"]

        def median_image_shift(other):
            shifts = []
            for y, x in spots_yx:
                y, x = int(y), int(x)
                if y < 50 or x < 50 or y >= h - 50 or x >= w - 50:
                    continue
                search = other[y-50:y+51, x-50:x+51]
                template = probe_raw[y-15:y+16, x-15:x+16]
                score = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                _, peak_score, _, loc = cv2.minMaxLoc(score)
                if peak_score > 0.08:
                    shifts.append((loc[1] - 35, loc[0] - 35))
            if len(shifts) < 8:
                raise ValueError("Could not match enough PSF spots across shifted probes")
            return np.median(np.asarray(shifts, np.float32), axis=0)

        x_path = root / "psf_probe_x_mean.npy"
        y_path = root / "psf_probe_y_mean.npy"
        if not x_path.exists() or not y_path.exists():
            raise ValueError("Capture PSF probes shifted horizontally and vertically")
        shift_yx_x = median_image_shift(np.load(x_path).astype(np.float32) - np.load(long_dark).astype(np.float32))
        shift_yx_y = median_image_shift(np.load(y_path).astype(np.float32) - np.load(long_dark).astype(np.float32))
        input_dx = float(offsets["x"][0])
        input_dy = float(offsets["y"][1])
        magnification_xy = (
            float(shift_yx_x[1] / input_dx),
            float(shift_yx_y[0] / input_dy),
        )
        # Keep the original response map for its diffuse component. The sparse
        # spot branch is mixed in separately according to its measured energy.
        probe = np.maximum(impulse - dark, 0)
        center_y, center_x = h // 2, w // 2
        print(
            f"Detected {len(spots_yx)} PSF spots; magnification "
            f"x={magnification_xy[0]:.4f}, y={magnification_xy[1]:.4f}"
        )
    else:
        # Backward-compatible path for the initial single-frame probe.
        probe = np.maximum(impulse - dark, 0)
        center_y, center_x = np.array(probe.shape) // 2

    # Keep the smallest centered region containing 99.9% of the PSF energy.
    cy, cx = center_y, center_x
    total = float(probe.sum())
    if total <= 0:
        raise ValueError("Impulse probe produced no signal above background")
    radius_y, radius_x = cy, cx
    prefix = np.pad(probe, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for radius in range(4, max(probe.shape) // 2, 4):
        y0, y1 = max(0, cy-radius), min(probe.shape[0], cy+radius+1)
        x0, x1 = max(0, cx-radius), min(probe.shape[1], cx+radius+1)
        enclosed = (prefix[y1, x1] - prefix[y0, x1]
                    - prefix[y1, x0] + prefix[y0, x0])
        if float(enclosed) / total >= 0.999:
            radius_y = min(radius, cy, probe.shape[0] - cy - 1)
            radius_x = min(radius, cx, probe.shape[1] - cx - 1)
            break
    psf = probe[cy-radius_y:cy+radius_y+1, cx-radius_x:cx+radius_x+1]
    psf_sum = float(psf.sum())
    if psf_sum <= 0:
        raise ValueError("Could not extract a centered impulse response")
    psf /= psf_sum

    # Uniform-field RAW response is the per-pixel gain map. A normalized PSF
    # preserves a uniform input, so this directly predicts the flat-field frame.
    flat_gain = np.maximum(flat255 - dark, 0)
    model = ForwardModel(
        dark=dark, flat_gain=flat_gain, psf=psf.astype(np.float32),
        response=response, exposure_us=int(metadata['exposure_us']),
        gain=int(metadata['gain']),
        display_shape=(int(metadata['screen_height']), int(metadata['screen_width'])),
        spots_yx=spots_yx, spot_weights=spot_weights,
        magnification_xy=magnification_xy,
        spot_mix=spot_mix,
    )
    model.save(output)
    return model


def render_display_canvas(
    image: np.ndarray, screen_shape: tuple[int, int], scale: float = 0.9,
    channel: str = "green",
) -> np.ndarray:
    """Render a source image into the canvas used by capture_imgnet.py."""
    from cv2 import INTER_AREA, resize

    src = np.asarray(image)
    if src.ndim == 2:
        gray = src
    elif src.ndim == 3 and src.shape[2] >= 3:
        index = {"r": 0, "red": 0, "g": 1, "green": 1,
                 "b": 2, "blue": 2}.get(channel.lower())
        if index is None:
            raise ValueError(f"Unsupported display channel: {channel}")
        gray = src[..., index]
    else:
        raise ValueError("image must be HxW grayscale or RGB")
    if gray.dtype != np.uint8:
        raise ValueError("image must contain uint8 values")
    sh, sw = screen_shape
    h, w = gray.shape
    factor = scale * min(sw / w, sh / h)
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    resized = resize(gray, (nw, nh), interpolation=INTER_AREA)
    canvas = np.zeros((sh, sw), np.uint8)
    x, y = (sw - nw) // 2, (sh - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit a forward model from calibration captures")
    parser.add_argument("calibration", nargs="?", default="forward_calibration")
    parser.add_argument("--output", default="forward_model.npz")
    args = parser.parse_args()
    fitted = fit_calibration(args.calibration, args.output)
    print(f"Saved {args.output}: PSF={fitted.psf.shape}, sensor={fitted.dark.shape}")
