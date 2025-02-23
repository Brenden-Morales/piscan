from socket_utils import SocketUtils
from camera_utils import CameraUtils
import socket
from picamera2 import Picamera2

picam2 = None

def start_server(host):
    port = 65432
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = SocketUtils.receive_message(conn)
                    if not data:
                        break
                    print("received message: {}".format(data.decode()))
                    SocketUtils.send_message(conn, data)  # Echo the received data back
    except KeyboardInterrupt:
        print("\nServer interrupted by user. Shutting down.")
    finally:
        s.close()
        print("Socket closed.")
        picam2.stop()
        print("picam stopped")

if __name__ == '__main__':
    camera_config = CameraUtils.getCameraConfig("camera_configs.yaml")
    camera_transform = CameraUtils.getCameraTransform(camera_config)
    Picamera2.set_logging(Picamera2.ERROR)
    picam2 = Picamera2()
    picam2.set_logging(Picamera2.ERROR)
    camera_config = picam2.create_still_configuration(transform=camera_transform)
    picam2.configure(camera_config)
    picam2.start()
    start_server(socket.gethostname() + ".local")