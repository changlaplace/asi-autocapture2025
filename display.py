import screeninfo
import cv2
import numpy as np
import screeninfo
import time
import threading
import os
from multiprocessing import Process, Value
import psutil


def display_image(image, scale, x_shift, y_shift, full_screen, display_flag, **kwargs):

    father_pid = kwargs.get('PID', None)

    display_flag.value = False
    screenid = 1 #second monitor
    screen = screeninfo.get_monitors()[screenid]

    if full_screen:
        s_w, s_h = screen.width, screen.height
        i_w, i_h, _ = np.shape(image)
        wx = float(s_w)/i_w
        hx = float(s_h)/i_h
        new_scale = min(wx, hx)
        scale = new_scale * scale

    #rotate image
    image = np.transpose(image, axes=(1, 0, 2))

    image_compound = np.zeros((screen.width, screen.height, 3), dtype='uint8')

    window_name = 'projector'
    #cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, screen.x + 1, screen.y + 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN)

    print("Screen size: ({}, {})".format(screen.width, screen.height))

    nx, ny, depth = image.shape
    print("Imported image width: {}".format(nx))
    print("Imported image height: {}".format(ny))

    nx_new, ny_new = int(scale*nx), int(scale*ny)
    print("Scaled image width: {}".format(nx_new))
    print("Scaled image height: {}".format(ny_new))
    image = cv2.resize(image, (ny_new, nx_new))

    #To make the image at the center
    x_offset = int(screen.width*0.5 - nx_new/2) + x_shift
    y_offset = int(screen.height*0.5 - ny_new/2) + y_shift

    assert ny_new <= screen.height and nx_new <= screen.width, "Image is too big for screen"

    print("Slicing {}:{}, {}:{}".format(x_offset, (nx_new + x_offset), y_offset ,(ny_new + y_offset)))
    image_compound[x_offset:(nx_new + x_offset), y_offset:(ny_new + y_offset), :] = image

    image_compound = np.transpose(image_compound, axes=(1, 0, 2))
    cv2.imshow(window_name, image_compound)

    display_flag.value = True

    time_resolution = 100 #milliseconds

    while(display_flag.value):
        cv2.waitKey(time_resolution) #display image for % milliseconds
        if father_pid is not None and not psutil.pid_exists(father_pid):
            print("Parent process with PID {} does not exist. Exiting display thread.".format(father_pid))
            break
    cv2.destroyAllWindows()
    return
    
class Display():
    def __init__(self):
        self.display_proc = None
        self.display_flag = Value('b', False)

    def start_display(self, image, scale, full_screen=False, x_shift=0, y_shift=0):
        print("Starting display thread")
        self.stop_display() #this prevents more than a single display thread from running
        print("Display thread stopped")
        self.display_proc = Process(target=display_image, 
                                    args=(image, 
                                          scale, 
                                          x_shift, 
                                          y_shift, 
                                          full_screen, 
                                          self.display_flag), 
                                          kwargs={'PID': os.getpid()})

        self.display_proc.start()

    def stop_display(self):
        self.display_flag.value = False
        time.sleep(0.1)
        if self.display_proc is not None:
            if self.display_proc.is_alive():
                self.display_proc.terminate()
        self.display_proc = None

    def close(self):
        self.stop_display()


if __name__=="__main__":
    random_image = np.random.randint(0, 255, (200, 200, 3), dtype='uint8')
    disp = Display()
    disp.start_display(random_image, scale=0.5, full_screen=True, x_shift=0, y_shift=0)
    time.sleep(5)  # Display for 5 seconds
    disp.stop_display()
    disp.close()
