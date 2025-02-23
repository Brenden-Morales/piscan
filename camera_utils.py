import socket
from yaml import safe_load, YAMLError
from libcamera import Transform

class CameraUtils:
    @staticmethod
    def getCameraConfig(filename):
        host_name = socket.gethostname()
        camera_config_dict = None
        with open(filename) as stream:
            try:
                configs = safe_load(stream)
                for config in configs:
                    if config['host'] == host_name:
                        camera_config_dict = config
            except YAMLError as exc:
                print(exc)
        if camera_config_dict is None:
            raise ValueError("No config found for {}".format(host_name))
        return camera_config_dict

    @staticmethod
    def getCameraTransform(camera_config_dict):
        camera_transform = Transform()
        if "rotation" in camera_config_dict:
            if camera_config_dict["rotation"] == 180:
                print("Rotating 180")
                camera_transform = Transform(hflip=1, vflip=1)
            else:
                raise ValueError("Rotation {} not supported".format(camera_config_dict["rotation"]))
        return camera_transform