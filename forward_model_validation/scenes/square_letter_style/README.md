# Square letter-style scene experiment

This run uses 20 distinct scenes in `assets_scenes/square_1600/`. They were center-cropped/resized to 1600 x 1600 RGB, matching the letter assets, then presented with the same green-only 90% fullscreen path. Captures use the calibrated RAW16 settings: 80 ms exposure, gain 0. Each `*_actual_raw16.npy` is a 1216 x 1936 uint16 camera frame. The corresponding `*_forward_comparison.png` compares that frame to the model prediction.

`reconstructions/` contains the spot-fold reconstructions from the RAW16 measurement. Reference images appear only in the evaluation panels; reconstruction itself uses the measurement. `reconstruction_montage.jpg` is the 20-scene overview.

## Result

Across the 20 frames, forward-model spatial correlation averaged 0.949 (range 0.880–0.985). Relative RMSE averaged 0.838 of mean signal (range 0.596–1.175). These correlations indicate the model captures a broad illumination envelope and scene-dependent pattern, but the errors are still large; correlation alone overstates pixel-level accuracy.

Matching square dimensions and the letter display scale did not make the existing spot-fold reconstruction generally successful. A few coarse shapes are visible, but most photos are only partly recovered and fine detail is lost. This points to a limitation in the current reconstruction/model’s spatial mixing and noise handling, rather than merely a mismatch in source aspect ratio. The square test also confirms that capture presentation is now controlled consistently with the letter calibration.

Run another capture with:

```powershell
.\.venv\Scripts\python.exe visualize_forward_model.py --asset assets_scenes\square_1600\scene_01_waterfall.png --scale 0.9
```

Reconstruct/evaluate a saved measurement with:

```powershell
.\.venv\Scripts\python.exe reconstruct_spot_array.py <measurement.npy> --reference <square-scene.png> --scale 0.9
```
