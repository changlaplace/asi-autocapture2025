import sys
sys.path.append(r"../End2endONN")
from get_dataset import get_dataset
# from pympler import tracker
import os
# from tools import get_image_files, mkdir_no_overwrite, save_as_16bit_tiff
from display import Display
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from asi_camera import Camera
# import zwoasi as asi

# asi.init(r"C:\ASI_Camera_SDK\ASI_Windows_SDK_V1.37\ASI SDK\lib\x64\ASICamera2.dll")

# camera = asi.Camera(0)
# asi._open_camera(0)

# camera_info = camera.get_camera_property()
# controls = camera.get_controls()


DEBUG = False


dataset_name = 'imagenet30'
dataset_rootdir = r"../End2endONN/data"
train_dataset, test_dataset = get_dataset(dataset_name, data_root=dataset_rootdir, download=True, resize=[224, 224])
camera = Camera(camera_id=0)  # Initialize the camera


def aquire_slidshow_dataset(dataset, pathname_out, number=-1, start_index=0):
    """ Aquire images from a dataset and save them in a specified directory.
    Args:
        dataset: The dataset to acquire images from.
        pathname_out: The output directory where images will be saved.
        number: The number of images to acquire. If -1, acquire all images.
        start_index: The index to start acquiring images from.
    """

    gain_dB = [0]#, 34, 37, 40] # gain in dB
    exposure_time_s = [1.5] #[1, 2, 4, 8, 16] # exposure in seconds
    sizes = [0.50] # size of the displayed image. There's a littlbe bug so for now 0.9 is the largest display size
    gammas = [1]#, 0.8, 0.9, 1
    frames = 1


    # At 0.9 corresponds to 2 and 7/16 inches on screen. For HFOV on diagonal of 20 deg need screen distance of 4.74 inches
    # Display seems to be 120 mm x 67.5 mm, 0.0625 mm pixels
    # Set to True to display images on the screen with plt.imshow()
    display = False

    if number < 0:
        number = len(dataset)
    disp = Display()
    for j in range(frames):
        for i, (image_file, image_label) in enumerate(dataset):
            if i < start_index:
                continue
            for scale in sizes:
                for exposure in exposure_time_s:
                    for gain in gain_dB:
                        for gamma in gammas:

                            disp_image = image_file.permute(1, 2, 0).numpy()  # Convert to HWC format
                            max_initial = np.max(disp_image)
                            disp_image = disp_image / max_initial  # Normalize to [0, 1] range
    
                            disp_image = disp_image ** gamma
                            disp_image = 255.0 * disp_image
                            # #disp_image = cv2.cvtColor(disp_image, cv2.COOR_BGR2RGB) Green only
                            # new_disp_image = np.zeros_like(disp_image)  Green only
                            # new_disp_image[:,:,1] = np.array(disp_image[:,:,1]) Green only
                            # disp_image = new_disp_image Green only

                            disp.start_display(disp_image, scale=scale, full_screen=True, y_shift = 0, x_shift = 0)
                            out_file = os.path.join(pathname_out, dataset_name + '_' + str(i) + '_' + str(image_label) + 'label')
                            file_ext = '_' + str(scale) + 'scale_' + str(gamma) + 'gamma_' + str(exposure) + 's_' + str(gain) + 'dB_frame' + str(j) + '_'
                            # props = {'display image': disp_image, 'output file': out_file}
                            # cam_image = Camera.take_picture(
                            #     gain_dB=gain,
                            #     exposure_time_s=exposure,
                            #     props=props,
                            #     bitdepth=Bitdepth.TWELVE)
                            captured = camera.capture_image(filefolder=r'./try_capture',
                                                            filename=r'dummy_capture.jpg', 
                                                            exposure = exposure, 
                                                            gain = gain
                                                            )

                            captured = captured.copy()

                            

                            if display:
                                plt.show()

        disp.close()

if __name__=="__main__":
    # This is the output image directory
    aquire_slidshow_dataset(pathname_out=r"C:\Minho\carvana_Captured_Images_training_Chip2", number=-1)
    print("Done!")
    exit()
