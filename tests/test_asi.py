import argparse
import os
import sys
import time

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


def save_control_values(filename, settings):
    filename += '.txt'
    with open(filename, 'w') as f:
        for k in sorted(settings.keys()):
            f.write('%s: %s\n' % (k, str(settings[k])))
    print('Camera settings saved to %s' % filename)

if not os.path.isfile(SDK_PATH):
    raise FileNotFoundError(
        f'ASI SDK DLL not found: {SDK_PATH}. Set ZWO_ASI_LIB to ASICamera2.dll.'
    )

asi.init(SDK_PATH)
camera = asi.Camera(0)

try:
    camera_info = camera.get_camera_property()
    controls = camera.get_controls()

    camera.stop_video_capture()
    camera.stop_exposure()
    camera.disable_dark_subtract()
    camera.set_control_value(
        asi.ASI_BANDWIDTHOVERLOAD,
        controls['BandWidth']['MinValue'],
    )
    camera.set_control_value(asi.ASI_GAIN, 0)
    camera.set_control_value(asi.ASI_EXPOSURE, 100)

    if camera_info['IsColorCam']:
        filename = 'image_color.jpg'
        image_type = asi.ASI_IMG_RGB24
        description = 'color'
    else:
        filename = 'image_mono.jpg'
        image_type = asi.ASI_IMG_RAW8
        description = '8-bit mono'

    camera.set_image_type(image_type)
    print(f'Capturing a single, {description} image')
    try:
        camera.capture(filename=filename)
    except asi.ZWO_CaptureError as exc:
        usb_mode = 'USB 3' if camera_info['IsUSB3Host'] else 'USB 2'
        raise RuntimeError(
            f'Exposure failed (status={exc.exposure_status}, connected via {usb_mode}). '
            'Reconnect the camera directly to a USB 3 port with a USB 3 data cable, '
            'then close any other camera software and retry.'
        ) from exc
    print('Saved to %s' % filename)
finally:
    camera.close()

