from picamera2 import Picamera2
from yaml import safe_load, YAMLError
import socket
from libcamera import Transform

host_name = socket.gethostname()

camera_config_dict = None
with open("camera_configs.yaml") as stream:
    try:
        configs = safe_load(stream)
        for config in configs:
            if config['host'] == host_name:
                camera_config_dict = config
    except YAMLError as exc:
        print(exc)

if camera_config_dict is None:
    raise ValueError("No config found for {}".format(host_name))

print(camera_config_dict)
camera_transform = Transform()
if "rotation" in camera_config_dict:
    if camera_config_dict["rotation"] == 180:
        print("Rotating 180")
        camera_transform = Transform(hflip=1, vflip=1)
    else:
        raise ValueError("Rotation {} not supported".format(camera_config_dict["rotation"]))

Picamera2.set_logging(Picamera2.ERROR)
picam2 = Picamera2()
picam2.set_logging(Picamera2.ERROR)
camera_config = picam2.create_still_configuration(transform=camera_transform)
picam2.configure(camera_config)
picam2.start()

picam2.capture_file("image.jpg")