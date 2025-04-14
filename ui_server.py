from PyInquirer import prompt, Separator
from cli_prompts import CliPrompts
from camera_controller import CameraController
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import concurrent.futures

camera_controllers = []
def create_camera_controllers(hosts):
    controllers = []
    for host in hosts:
        hostname = host + ".local"
        controllers.append(CameraController(hostname, 65432))
    return controllers

if not os.path.exists("./captures"):
    os.makedirs("./captures")

selected_hosts = prompt(CliPrompts.get_hosts_prompt("camera_configs.yaml"))
camera_controllers = create_camera_controllers(selected_hosts["Hosts"])
for camera_controller in camera_controllers:
    print('Homogenizing controls')
    camera_controller.set_controls(camera_controllers[0].controls)

class CameraSettings(BaseModel):
    AfMode: int
    AfTrigger: int
    AnalogueGain: int
    AwbMode: int
    ExposureTime: int
    AfRange: int
    AfSpeed: int

app = FastAPI()
# Serve static files (like index.html, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Simple API endpoint
@app.get("/api/greet")
def greet():
    return {"message": "Hello from the API!"}

# Simple API endpoint
@app.put("/api/settings")
def update_settings(settings: CameraSettings):
    print(settings)
    for camera_controller in camera_controllers:
        camera_controller.controls['AwbMode'][-1] = settings.AwbMode
        camera_controller.controls['AfMode'][-1] = settings.AfMode
        camera_controller.controls['AfTrigger'][-1] = settings.AfTrigger
        camera_controller.controls['AfRange'][-1] = settings.AfRange
        camera_controller.controls['AfSpeed'][-1] = settings.AfSpeed
        camera_controller.controls['AnalogueGain'][-1] = settings.AnalogueGain
        camera_controller.controls['ExposureTime'][-1] = settings.ExposureTime

        camera_controller.controls['AnalogueGain'][-1] = settings.AnalogueGain
        camera_controller.set_controls(camera_controller.controls)
    return {"message": "Hello from the API!"}

@app.put("/api/snap")
def update_settings():
    print("snap")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for camera_controller in camera_controllers:
            futures.append(executor.submit(camera_controller.take_snap, False, "static/{}_snap.jpg".format(camera_controller.host)))
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return {"message": "SUCCESS"}

@app.put("/api/capture")
def update_settings():
    print("capture")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for camera_controller in camera_controllers:
            futures.append(executor.submit(camera_controller.take_snap, True, "captures/{}_snap.jpg" ))
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return {"message": "SUCCESS"}

# UI route
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r") as f:
        return f.read()