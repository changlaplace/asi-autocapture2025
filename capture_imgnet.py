
def aquire_slidshow_dataset(dataset, pathname_out, number=-1, start_index=0):
    """ Aquire images from a dataset and save them in a specified directory.
    Args:
        dataset: The dataset to acquire images from.
        pathname_out: The output directory where images will be saved.
        number: The number of images to acquire. If -1, acquire all images.
        start_index: The index to start acquiring images from.
    """
    gain_dB = [0]#, 34, 37, 40] # gain in dB
    exposure_time_s = [0.12] #[1, 2, 4, 8, 16] # exposure in seconds
    sizes = [0.50] # size of the displayed image. There's a littlbe bug so for now 0.9 is the largest display size
    gammas = [1]#, 0.8, 0.9, 1
    frames = 1
    
    # At 0.9 corresponds to 2 and 7/16 inches on screen. For HFOV on diagonal of 20 deg need screen distance of 4.74 inches
    # Display seems to be 120 mm x 67.5 mm, 0.0625 mm pixels
    # Set to True to display images on the screen with plt.imshow()
    os.makedirs(pathname_out, exist_ok=True)
    display = False
    global Camera
    global logger
    if MP_save:
        global pool
        global sem

    if number < 0:
        number = len(dataset)
    disp = Display()
    for j in range(frames):
        for i in range(0, number):
            if i < start_index:
                logger.info(f"Skipping the {i} th image in the total {number} to take")
                continue
            image_file = dataset[i][0]
            image_label = dataset[i][1]
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
                            logger.info(f"Creating a display window for the {i} th image in total {number} images")
                            disp.start_display(disp_image, scale=scale, full_screen=True, y_shift = 0, x_shift = 0)
                            while not disp.display_flag.value:
                                time.sleep(0.1)
                                logger.info("Waiting for display to start...")
                            time.sleep(0.1)
                            break_patience = 5
                            break_count = 0
                            while True:
                                try:
                                    captured = Camera.capture_image(exposure = exposure, 
                                                            gain = gain,
                                                            set_image_type=asi.ASI_IMG_RGB24
                                                            )
                                    if captured.max() > 253:
                                        exposure *= 0.8
                                        logger.info(f'Image saturation detected, max value: {captured.max()}, \
                                              decreasing exposure to {exposure}s')
                                    else:
                                        break
                                except Exception as e:
                                    break_count += 1
                                    if break_count > break_patience:
                                        logger.info(f'Maximum break count exceeded: {break_count}. Exiting...')
                                        sys.exit(f"Failed to capture image after {break_patience} attempts: {e}")
                                    logger.info(f"Error capturing image: {e}, with break count {break_count}. Retrying after a miniute...")
                                    time.sleep(60)
                                    Camera = ASICamera(camera_id=0)  # Reinitialize the camera

                            captured_copy = captured.copy()                            
                            out_file = dataset_name + '_' + str(i) + '_' + str(image_label) + 'label' + \
                                '_' + str(scale) + 'scale_' + str(gamma) + 'gamma_' + str(exposure) + 's_' + \
                                str(gain) + 'dB_frame' + str(j) + '.png'
                            try:
                                logger.info(f"Saveing the image into {os.path.join(pathname_out, out_file)}")
                                if MP_save:
                                    sem.acquire()
                                    pool.apply_async(save_image, args=(
                                        captured_copy, 
                                        os.path.join(pathname_out, out_file), 
                                        asi.ASI_IMG_RGB24
                                    ),
                                    callback=lambda _:sem.release())
                                else: 
                                    Camera.save_captured_image(captured_copy, 
                                                        os.path.join(pathname_out, out_file), 
                                                        set_image_type=asi.ASI_IMG_RGB24, sem=sem)
                            except Exception as e:
                                logger.info(f"Failed in saving the image, continue")
                            

                            if display:
                                plt.imshow(captured_copy)
                                plt.show()

        disp.close()

if __name__=="__main__":
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
    from asi_camera import ASICamera
    import zwoasi as asi
    from utlis import get_current_captured_number, setup_logger, save_image
    from multiprocessing import Pool, Semaphore
    import multiprocessing as mp

    MP_save = True ### If we save images with pool multiprocess
    NUM_workers = 20
    if MP_save:
        pool = mp.Pool(NUM_workers)
        sem = Semaphore(4)


    dataset_name = 'imagenet30'
    dataset_rootdir = r"../End2endONN/data"
    train_dataset, test_dataset = get_dataset(dataset_name, data_root=dataset_rootdir, download=True, resize=[224, 224])

    # logger = setup_logger(r'./capturing_logs')
    logger = setup_logger(r'C:/yuboz4/launcher_logs')
    logger.info('Finished setting up logging')
    
    Camera = ASICamera(camera_id=0)  # Initialize the camera
    # This is the output image directory
    logger.info(f"Preparing to take images for train dataset")
    train_save_folder = r"C:\yuboz4\Imagenet_data\Hyperbolid\train"
    train_start_num = get_current_captured_number(train_save_folder)
    aquire_slidshow_dataset(train_dataset, pathname_out=train_save_folder, number=-1, start_index=train_start_num)

    logger.info(f"Preparing to take images for test dataset")
    test_save_folder = r"C:\yuboz4\Imagenet_data\Hyperbolid\test"
    test_start_num = get_current_captured_number(test_save_folder)
    aquire_slidshow_dataset(test_dataset, pathname_out=test_save_folder, number=-1, start_index=test_start_num)
    print("Done!")
    exit()
