import os

SDK_PATH = os.environ.get(
    'ZWO_ASI_LIB',
    r'C:\Program Files\ASIStudio\ASICamera2.dll',
)
os.environ.setdefault('ZWO_ASI_LIB', SDK_PATH)
SDK_DIRECTORY = os.path.dirname(SDK_PATH)
_SDK_DLL_DIRECTORY = None
if os.path.isdir(SDK_DIRECTORY):
    os.environ['PATH'] = SDK_DIRECTORY + os.pathsep + os.environ['PATH']
    _SDK_DLL_DIRECTORY = os.add_dll_directory(SDK_DIRECTORY)

import zwoasi as asi
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class ASICamera:
    def __init__(self, camera_id=0):
        if not os.path.isfile(SDK_PATH):
            raise FileNotFoundError(
                f'ASI SDK DLL not found: {SDK_PATH}. '
                'Set ZWO_ASI_LIB to ASICamera2.dll.'
            )
        asi.init(SDK_PATH)
        try:
            self.camera = asi.Camera(camera_id)
            self.camera_info = self.camera.get_camera_property()
            self.controls = self.camera.get_controls()
            self.camera.set_control_value(
                asi.ASI_BANDWIDTHOVERLOAD,
                self.controls['BandWidth']['MinValue'],
            )
            self.camera.disable_dark_subtract()
            for stop_capture in (
                self.camera.stop_video_capture,
                self.camera.stop_exposure,
            ):
                try:
                    stop_capture()
                except asi.ZWO_Error:
                    pass
        except Exception as e:
            print(f"Error initializing camera with ID {camera_id}: {e}")
            raise


    def capture_image(self, exposure=1.0, gain=0, set_image_type=asi.ASI_IMG_RAW8):
        """Capture a single image with specified exposure and gain.
        Args:
            filefolder (str): Directory to save the captured image.
            filename (str): Name of the file to save the image.
            exposure (float): Exposure time in seconds.
            gain (int): Gain value.
            set_image_type: Image type to set for the camera.
        """

        if set_image_type not in self.camera_info['SupportedVideoFormat']:
            raise ValueError(
                f'Image type {set_image_type} is not supported by '
                f'{self.camera_info["Name"]}; supported types are '
                f'{self.camera_info["SupportedVideoFormat"]}.'
            )

        exposure = round(exposure * 1e6)
        self.camera.set_control_value(asi.ASI_GAIN, gain) #copied from example
        self.camera.set_control_value(asi.ASI_EXPOSURE, exposure) # microseconds
        self.camera.set_image_type(set_image_type)
        try:
            captured_img = self.camera.capture()
        except asi.ZWO_CaptureError as exc:
            usb_mode = 'USB 3' if self.camera_info['IsUSB3Host'] else 'USB 2'
            raise RuntimeError(
                f'Exposure failed (status={exc.exposure_status}, connected via '
                f'{usb_mode}). Reconnect the camera directly to a USB 3 port '
                'with a USB 3 data cable and close other camera software.'
            ) from exc
        return captured_img

    def close(self):
        """Release the camera so another process can open it."""
        if getattr(self, 'camera', None) is not None:
            self.camera.close()
            self.camera = None

    def save_captured_image(self, img, filename, set_image_type=asi.ASI_IMG_RAW8):
        """Save the captured image to a file."""
        if filename is not None:
            mode = None
            if len(img.shape) == 3:
                img = img[:, :, ::-1]  # Convert BGR to RGB
            if set_image_type == asi.ASI_IMG_RAW16:
                mode = 'I;16'
            image = Image.fromarray(img, mode=mode)
            image.save(filename)
        return 

        
    def save_control_values(self, filename):
        """Save camera control settings to a text file."""
        settings = self.camera.get_control_values()
        filename += '.txt'
        with open(filename, 'w') as f:
            for k in sorted(settings.keys()):
                f.write('%s: %s\n' % (k, str(settings[k])))
        print('Camera settings saved to %s' % filename)

if __name__ == "__main__":
    
    camera = ASICamera(camera_id=0)  # Initialize the camera
    captured_img = camera.capture_image(exposure=1, gain=0, set_image_type=asi.ASI_IMG_RAW16)
    camera.save_captured_image(captured_img, filename='captured_image.tiff', set_image_type=asi.ASI_IMG_RAW16)
    print(np.max(captured_img))
    print(captured_img.shape)
    plt.figure()
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(captured_img.ravel())
    plt.show()
    read = cv2.imread('captured_image.tiff', cv2.IMREAD_UNCHANGED)
    print(read.max())
    print("Image capture complete.")
