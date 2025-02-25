from socket_utils import SocketUtils
from camera_utils import CameraUtils
import socket
from picamera2 import Picamera2
import traceback
import json
import time

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
                    message = data.decode()
                    print("received message: {}".format(message))
                    if message == SocketUtils.TAKE_SNAPSHOT:
                        picam2.capture_file("snap.jpg")
                        with open("snap.jpg", 'rb') as file:
                            byte_array = bytearray(file.read())
                        print('Captured JPEG Buffer')
                        SocketUtils.send_message(conn, byte_array)
                        print('Sent JPEG Buffer')
                    elif message == SocketUtils.GET_CONTROLS:
                        SocketUtils.send_message(conn, json.dumps(picam2.camera_controls).encode())
                        print('Sent controls')
                    elif message.startswith(SocketUtils.SET_CONTROLS):
                        print('SETTING CONTROLS')
                        controls = message.removeprefix(SocketUtils.SET_CONTROLS)
                        controls = json.loads(controls)
                        for control_name, control_value in controls.items():
                            control_obj = {control_name: control_value}
                            print(control_obj)
                            picam2.set_controls(control_obj)
                        SocketUtils.send_message(conn, "controls set".encode())
                    else:
                        SocketUtils.send_message(conn, data)  # Echo the received data back
    except KeyboardInterrupt:
        print("\nServer interrupted by user. Shutting down.")
        print(traceback.format_exc())
    except BrokenPipeError:
        print("Broken pipe: The client may have disconnected before receiving all the data.")
        print(traceback.format_exc())
    finally:
        s.close()
        print("Socket closed.")
        picam2.stop()
        print("picam stopped")
        traceback.print_exc()

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