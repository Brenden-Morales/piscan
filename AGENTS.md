# Code for experimenting with SLAM and photogrammetry

The point of this project is to experiment and learn with photogrammetry and SLAM.

There is a camera rig that consists of six synchronized cameras. The "boot_camera_servers.sh" file will boot the cameras
on remote machines and then start two UI servers. One is for svelte, one is for interacting with a python backend.

Once images have been captured from the rig they go into the "captures" directory. 

Then "intrinsics_calibration.py" can be run to get camera intrinsics

Then "stereo_calibration.py" can be run to calculate the stereo relationships of cameras

Finally, "bundle_adjustment.py" can be run.

This is a sequential process that depends on the output of the previous script.


Make sure any python commands are run with `pipenv run`