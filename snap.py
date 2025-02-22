from picamera2 import Picamera2, Preview
import time
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