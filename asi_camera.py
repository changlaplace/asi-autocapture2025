import zwoasi as asi
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class ASICamera:
    def __init__(self, camera_id=0):
        
        asi.init(r"C:\ASI_Camera_SDK\ASI_Windows_SDK_V1.37\ASI SDK\lib\x64\ASICamera2.dll")
        try:
            self.camera = asi.Camera(camera_id)
            self.camera_info = self.camera.get_camera_property()
            self.controls = self.camera.get_controls()
        except Exception as e:
            print(f"Error initializing camera with ID {camera_id}: {e}")
            raise


    def capture_image(self, exposure=1e6, gain=0, set_image_type=asi.ASI_IMG_RGB24):
        """Capture a single image with specified exposure and gain.
        Args:
            filefolder (str): Directory to save the captured image.
            filename (str): Name of the file to save the image.
            exposure (float): Exposure time in seconds.
            gain (int): Gain value.
            set_image_type: Image type to set for the camera.
        """

        exposure = round(exposure*1e6)
        self.camera.set_control_value(asi.ASI_GAIN, gain) #copied from example
        self.camera.set_control_value(asi.ASI_EXPOSURE, exposure) # microseconds
        self.camera.set_image_type(set_image_type)
        captured_img = self.camera.capture()
        return captured_img
    def save_captured_image(self, img, filename, set_image_type=asi.ASI_IMG_RGB24):
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
