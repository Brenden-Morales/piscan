import socket
import json
from socket_utils import SocketUtils
import os

ENABLED_CONTROLS = ["AnalogueGain", "ExposureTime", 'AwbMode']

class CameraController:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.controls = self.get_controls()


    def send_then_receive(self, message):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(30)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            sock.connect((self.host, self.port))
            SocketUtils.send_message(sock, message.encode())
            print("Sent {} to camera server {}:{}".format(message, self.host, self.port))
            return SocketUtils.receive_message(sock)

    def get_controls(self):
        print('Getting controls')
        controls_string = self.send_then_receive(SocketUtils.GET_CONTROLS).decode()
        print("Received controls from camera server {}:{}".format(self.host, self.port))
        controls = json.loads(controls_string)
        good_controls = {}
        for control_name, control_values in controls.items():
            if control_name in ENABLED_CONTROLS:
                good_controls[control_name] = control_values
        return good_controls

    def set_controls(self, controls):
        print('Setting controls')
        control_values = {k: v[-1] for k, v in controls.items()}
        set_controls_string = SocketUtils.SET_CONTROLS + json.dumps(control_values)
        set_controls_result = self.send_then_receive(set_controls_string).decode()
        print("Set controls to camera server {}:{}".format(self.host, self.port))

    def take_snap(self, useHistory = False):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            data = self.send_then_receive(SocketUtils.TAKE_SNAPSHOT)
            print("Received data from camera server {}:{}".format(self.host, self.port))
            file_path = "captures/{}_snap.jpg".format(self.host)
            if useHistory:
                project_dir = "captures/{}".format(self.host)
                dir_size = len(os.listdir(project_dir))
                file_path = "captures/{}/{}.jpg".format(self.host, dir_size)
            with open(file_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(data)
            print("File written to: {}".format(file_path))