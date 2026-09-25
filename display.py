import os
import time
from multiprocessing import Process, Value

import cv2
import numpy as np
import psutil
import screeninfo


def display_image(
    image,
    scale,
    x_shift,
    y_shift,
    full_screen,
    screen_id,
    display_flag,
    **kwargs,
):
    parent_pid = kwargs.get('PID')
    display_flag.value = False
    monitors = screeninfo.get_monitors()
    if screen_id < 0 or screen_id >= len(monitors):
        print(f'Invalid screen index {screen_id}; available screens: 0-{len(monitors) - 1}')
        return

    screen = monitors[screen_id]
    image_height, image_width = image.shape[:2]
    if full_screen:
        scale *= min(screen.width / image_width, screen.height / image_height)

    new_width = max(1, int(image_width * scale))
    new_height = max(1, int(image_height * scale))
    if new_width > screen.width or new_height > screen.height:
        raise ValueError('Scaled image is larger than the selected screen.')

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)
    x_offset = (screen.width - new_width) // 2 + x_shift
    y_offset = (screen.height - new_height) // 2 + y_shift
    if (
        x_offset < 0
        or y_offset < 0
        or x_offset + new_width > screen.width
        or y_offset + new_height > screen.height
    ):
        raise ValueError('The shifted image does not fit on the selected screen.')

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = image

    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, screen.x + 1, screen.y + 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)
    display_flag.value = True

    while display_flag.value:
        cv2.waitKey(100)
        if parent_pid is not None and not psutil.pid_exists(parent_pid):
            break

    cv2.destroyAllWindows()


class Display:
    def __init__(self, screen_id=1):
        self.screen_id = screen_id
        self.display_proc = None
        self.display_flag = Value('b', False)

    def start_display(self, image, scale, full_screen=False, x_shift=0, y_shift=0):
        self.stop_display()
        self.display_proc = Process(
            target=display_image,
            args=(
                image,
                scale,
                x_shift,
                y_shift,
                full_screen,
                self.screen_id,
                self.display_flag,
            ),
            kwargs={'PID': os.getpid()},
        )
        self.display_proc.start()

    def stop_display(self):
        self.display_flag.value = False
        if self.display_proc is not None:
            self.display_proc.join(timeout=0.5)
            if self.display_proc.is_alive():
                self.display_proc.terminate()
                self.display_proc.join(timeout=1)
        self.display_proc = None

    def close(self):
        self.stop_display()


if __name__ == '__main__':
    random_image = np.random.randint(0, 255, (1200, 1200, 3), dtype=np.uint8)
    display = Display(screen_id=1)
    display.start_display(random_image, scale=0.9, full_screen=True)
    time.sleep(5)
    display.close()
