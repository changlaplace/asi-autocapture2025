import zwoasi as asi
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

class ASICamera:
    def __init__(self, camera_id=0):
        
        asi.init(r"C:\ASI_Camera_SDK\ASI_Windows_SDK_V1.37\ASI SDK\lib\x64\ASICamera2.dll")
        self.camera = asi.Camera(camera_id)
        self.camera_info = self.camera.get_camera_property()
        self.controls = self.camera.get_controls()


    def capture_image(self, filefolder = r'.',filename=r'dummy_capture.jpg', exposure=1e6, gain=0, set_image_type=asi.ASI_IMG_RGB24):
        """Capture a single image with specified exposure and gain.
        Args:
            filefolder (str): Directory to save the captured image.
            filename (str): Name of the file to save the image.
            exposure (float): Exposure time in seconds.
            gain (int): Gain value.
            set_image_type: Image type to set for the camera.
        """

        if not os.path.exists(filefolder):
            os.makedirs(filefolder)
        exposure = round(exposure*1e6)
        self.camera.set_control_value(asi.ASI_GAIN, gain) #copied from example
        self.camera.set_control_value(asi.ASI_EXPOSURE, exposure) # microseconds
        self.camera.set_image_type(set_image_type)
        filename_to_save  = os.path.join(filefolder, filename)
        self.camera.capture(filename=filename_to_save)
        print(f'Captured image saved to {filename}')


if __name__ == "__main__":
    
    camera = ASICamera(camera_id=0)  # Initialize the camera
    camera.capture_image(filefolder=r'.', filename='captured_im23age.tiff', exposure=1, gain=0, set_image_type=asi.ASI_IMG_Y8)
    captured_img = cv2.imread('captured_im23age.tiff', cv2.IMREAD_UNCHANGED)
    print(np.max(captured_img))
    print(captured_img.shape)
    hist, bins = np.histogram(captured_img, bins=256, range=(0, 256))

    plt.figure()
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(captured_img.ravel())
    # plt.xlim([0, 256])
    plt.show()
    print("Image capture complete.")
