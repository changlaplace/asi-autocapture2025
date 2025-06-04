import argparse
import os
import sys
import time
import zwoasi as asi


def save_control_values(filename, settings):
    filename += '.txt'
    with open(filename, 'w') as f:
        for k in sorted(settings.keys()):
            f.write('%s: %s\n' % (k, str(settings[k])))
    print('Camera settings saved to %s' % filename)

asi.init(r"C:\ASI_Camera_SDK\ASI_Windows_SDK_V1.37\ASI SDK\lib\x64\ASICamera2.dll")

camera = asi.Camera(0)
asi._open_camera(0)

camera_info = camera.get_camera_property()
controls = camera.get_controls()

exposure=round(1e6) # in microseconds

gain=0

camera.set_control_value(asi.ASI_GAIN, gain) #copied from example
camera.set_control_value(asi.ASI_EXPOSURE, exposure) # microseconds


filename = 'image_color.jpg'
camera.set_image_type(asi.ASI_IMG_RGB24)
print('Capturing a single, color image')
camera.capture(filename=filename)
# print('Saved to %s' % filename)
# save_control_values(filename, camera.get_control_values())

