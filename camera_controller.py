import socket
from socket_utils import SocketUtils

class CameraController:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def take_snap(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            print(self.host)
            print(self.port)
            sock.connect((self.host, self.port))
            SocketUtils.send_message(sock, SocketUtils.TAKE_SNAPSHOT.encode())
            print("Sent {} to camera server {}:{}".format(SocketUtils.TAKE_SNAPSHOT, self.host, self.port))
            data = SocketUtils.receive_message(sock)
            print("Received data from camera server {}:{}".format(self.host, self.port))
