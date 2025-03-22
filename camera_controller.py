import socket
import json
from socket_utils import SocketUtils
import os

ENABLED_CONTROLS = ["AnalogueGain", "ExposureTime", 'AwbMode', 'AeEnable']

class CameraController:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.sock.settimeout(600)
        self.sock.setblocking(True)
        self.sock.connect((self.host, self.port))
        self.controls = self.get_good_controls()

    def send_then_receive(self, message):
        SocketUtils.send_message(self.sock, message.encode())
        print("Sent {} to camera server {}:{}".format(message, self.host, self.port))
        return SocketUtils.receive_message(self.sock)

    def get_all_controls(self):
        print('Getting all controls')
        controls_string = self.send_then_receive(SocketUtils.GET_CONTROLS).decode()
        print("Received controls from camera server {}:{}".format(self.host, self.port))
        controls = json.loads(controls_string)
        return controls

    def get_good_controls(self):
        controls = self.get_all_controls()
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
        data = self.send_then_receive(SocketUtils.TAKE_SNAPSHOT)
        print("Received data from camera server {}:{}".format(self.host, self.port))
        file_path = "captures/{}_snap.jpg".format(self.host)
        if useHistory:
            project_dir = "captures/{}".format(self.host)
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)
            dir_size = len(os.listdir(project_dir))
            file_path = "captures/{}/{}.jpg".format(self.host, dir_size)
        with open(file_path, "wb") as binary_file:
            # Write bytes to file
            binary_file.write(data)
        print("File written to: {}".format(file_path))

    def close_socket(self):
        print("closing socket to {}".format(self.host))
        self.sock.close()
