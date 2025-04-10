from socket_utils import SocketUtils
from camera_utils import CameraUtils
import socket
from picamera2 import Picamera2
import piexif
import traceback
import json
import time
import io

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
                            # exif read?
                            exif_dict = piexif.load(byte_array)
                            # Get the existing 'Model' tag (tag ID 272 in the 0th IFD)
                            model_tag = piexif.ImageIFD.Model
                            existing_model = exif_dict["0th"].get(model_tag, b"").decode("utf-8", errors="ignore")
                            # Append your custom string
                            new_model = existing_model + "_" + host
                            # Set the new value (must be bytes)
                            exif_dict["0th"][model_tag] = new_model.encode("utf-8")
                            # Dump the new EXIF data
                            exif_bytes = piexif.dump(exif_dict)
                            # Prepare a BytesIO buffer to receive the output
                            output_stream = io.BytesIO()
                            piexif.insert(exif_bytes, bytes(byte_array), output_stream)
                            # Get the modified image data as bytes or bytearray
                            new_image_bytes = output_stream.getvalue()
                            new_image_bytearray = bytearray(new_image_bytes)
                            print('Captured JPEG Buffer')
                            SocketUtils.send_message(conn, new_image_bytearray)
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
                        # picam2.set_controls({"ColourGains": (6, 6)})
                        picam2.set_controls({"AwbEnable": True})
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
    camera_config = picam2.create_still_configuration(transform=camera_transform, )
    picam2.configure(camera_config)
    picam2.start()
    start_server(socket.gethostname() + ".local")