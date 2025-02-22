from picamera2 import Picamera2
from yaml import safe_load, YAMLError
import socket

host_name = socket.gethostname()

camera_config = None
with open("camera_configs.yaml") as stream:
    try:
        configs = safe_load(stream)
        for config in configs:
            if config['host'] == host_name:
                camera_config = config
    except YAMLError as exc:
        print(exc)

if camera_config is None:
    raise ValueError("No config found for {}".format(host_name));

print(camera_config)


Picamera2.set_logging(Picamera2.ERROR)
picam2 = Picamera2()
picam2.set_logging(Picamera2.ERROR)
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)
picam2.start()
picam2.capture_file("image.jpg")
print("OHHHH SHIT")
print("HOW ABOUT NOW?")
print("asdf?")