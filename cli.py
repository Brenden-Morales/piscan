# -*- coding: utf-8 -*-
"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from PyInquirer import prompt, Separator
from cli_prompts import CliPrompts
from camera_controller import CameraController
import concurrent.futures

camera_controllers = []
def create_camera_controllers(hosts):
    controllers = []
    for host in hosts:
        hostname = host + ".local"
        controllers.append(CameraController(hostname, 65432))
    return controllers

selected_hosts = prompt(CliPrompts.get_hosts_prompt("camera_configs.yaml"))
camera_controllers = create_camera_controllers(selected_hosts["Hosts"])
for camera_controller in camera_controllers:
    camera_controller.set_controls(camera_controllers[0].controls)
op = ""
while op != "Quit":
    op = prompt(CliPrompts.get_operation_prompt())["operation"]
    if op == "Select Hosts":
        selected_hosts = prompt(CliPrompts.get_hosts_prompt("camera_configs.yaml"))
        camera_controllers = create_camera_controllers(selected_hosts["Hosts"])
    elif op == "Take snapshot":
        print("take a snapshot!")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for camera_controller in camera_controllers:
                futures.append(executor.submit(camera_controller.take_snap))
            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()
