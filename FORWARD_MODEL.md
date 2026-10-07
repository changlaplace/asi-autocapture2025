# Display-to-camera forward model

The camera sees a lenslet spot lattice. A single display point makes a 2D array
of spots, and shifting that point moves the image inside each spot cell. The
current model convolves the display image with the measured 2D spot-array PSF,
then adds a small contribution from a compact renderer for detected spot
centers. It also includes a measured background, flat-field gain, and display
intensity response curve.

## Calibration

The fitted model in `forward_model.npz` uses the ZWO ASI174MM, the 1920x1080
green display channel, RAW16, 80 ms exposure, and gain 0. The initial calibration
captures the black screen, ten intensity levels, a white field, and a 22x22
pixel probe:

```powershell
.\.venv\Scripts\python.exe calibrate_forward_model.py --screen 1 --exposure 0.08 --gain 0
```

Then acquire the repeated long-exposure spot array. This captures monitor-black
and center, horizontal-offset, and vertical-offset probes at 1 second, three
frames each. It does not show a full white field at that exposure:

```powershell
.\.venv\Scripts\python.exe capture_long_psf.py --screen 1 --exposure 1.0 --gain 0 --repeats 3
.\.venv\Scripts\python.exe forward_model.py forward_calibration --output forward_model.npz
```

Use the same display, screen geometry, green channel, camera exposure, and gain
for later measurements. The spot lattice diagnostic is
`forward_calibration/psf_shift_diagnostic.png`.

## Predict

```python
from PIL import Image
import numpy as np
from forward_model import ForwardModel, render_display_canvas

model = ForwardModel.load("forward_model.npz")
source_rgb = np.asarray(Image.open("assets_letters/01_A.png").convert("RGB"))
canvas = render_display_canvas(source_rgb, model.display_shape, scale=0.9, channel="green")
predicted_raw16 = model.predict(canvas)
```

`predicted_raw16` is float32 in RAW16 camera digital-number units. The prediction
uses an FFT convolution with the measured spot-array PSF and a small residual
spot-copy contribution. It does not model arbitrary lenslet-to-lenslet
distortion or clipping.
Monitor-black is used as the background and includes ambient light and black
level leakage; it is not a shutter-closed dark frame.

## Current letter check

The first rendering used the sign of the horizontal probe shift as an image
flip, which mirrored asymmetric letters such as B. The renderer now keeps the
camera-observed orientation. Five measured letters (A, B, C, D, O) were used to
fit shared spatial and intensity parameters. The PSF convolution now samples
its camera-plane input 1.30x wider horizontally and 1.00x vertically. A small
0.10 contribution from the spot-copy renderer remains. The renderer's stored
probe magnifications are -0.1169 and 0.1278; the negative x value records probe
motion and does not flip the image. A fitted scalar signal scale of 0.776
compensates for the brightness change from wider spatial sampling.

At the same RAW16 exposure and gain, prediction errors are A: RMSE 156 DN
(15.9% of mean light signal), r=0.993; B: 184 DN (14.6%), r=0.993; C: 160 DN
(17.7%), r=0.990; D: 169 DN (15.3%), r=0.992; O: 174 DN (15.6%), r=0.992.
F was captured after fitting as a held-out check: RMSE 158 DN (20.5%) and
r=0.987. The comparisons are saved as
`forward_model_validation/<letter>_comparison.png`; measured RAW16 frames are
saved next to them. Predicted arrays are recomputed on demand rather than
stored as duplicate multi-megabyte files. This is still a compact linear
approximation; it does not model lenslet-specific distortion or
intensity-dependent camera effects.
