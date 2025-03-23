# -*- coding: utf-8 -*-
"""
* Checkbox question example
* run example by typing `python example/checkbox.py` in your console
"""
from PyInquirer import prompt, Separator
from cli_prompts import CliPrompts
from camera_controller import CameraController
import concurrent.futures
import os
from pprint import pprint
from bt_utils import BTUtils
import time
import traceback

STEPS_PER_REV = 200
STEPS_PER_SNAP = 2
MILLISECONDS_PER_STEP = 200
EXPOSURE_TIME = 20000

if not os.path.exists("./captures"):
    os.makedirs("./captures")

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
    print('Homogenizing controls')
    camera_controller.set_controls(camera_controllers[0].controls)

# stepper = BTUtils()

op = ""
try:
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
        elif op == "Set Controls":
            control_name, control_value = CliPrompts.list_controls_prompt(camera_controllers[0].controls)
            if control_name is not None:
                for camera_controller in camera_controllers:
                    camera_controller.controls[control_name][-1] = control_value
                    camera_controller.set_controls(camera_controller.controls)
        elif op == "List Controls":
            print("List Controls!")
            combined_controls = {}
            for camera_controller in camera_controllers:
                all_camera_controls = camera_controller.get_all_controls()
                for control_name, control_value in all_camera_controls.items():
                    if control_name not in combined_controls:
                        combined_controls[control_name] = [control_value[-1]]
                    else:
                        combined_controls[control_name].append(control_value[-1])
                pprint(all_camera_controls, indent=4)
                print()
                print()
            print("All configs combined:")
            pprint(combined_controls, indent=4)

        elif op == "Start project":
            project_prompt = ""
            while project_prompt != "Quit":
                project_prompt = CliPrompts.single_list_prompt(["Snap", "Quit"])
                if project_prompt == "Snap":
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        futures = []
                        for camera_controller in camera_controllers:
                            futures.append(executor.submit(camera_controller.take_snap, True))
                        # Wait for all futures to complete
                        for future in concurrent.futures.as_completed(futures):
                            future.result()

        elif op == "MAGIC":
            print("START THE MAGIC")
            steps = 0
            while steps < STEPS_PER_REV:
                steps += STEPS_PER_SNAP
                stepper.step(STEPS_PER_SNAP)
                time.sleep((STEPS_PER_SNAP * MILLISECONDS_PER_STEP) / 1000)
                time.sleep(1)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = []
                    for camera_controller in camera_controllers:
                        futures.append(executor.submit(camera_controller.take_snap, True))
                    # Wait for all futures to complete
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
except:
    print("Broken pipe: The cli may have disconnected before receiving all the data.")
    print(traceback.format_exc())
finally:
    print("closing camera controllers")
    for camera_controller in camera_controllers:
        camera_controller.close_socket()