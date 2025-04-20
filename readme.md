ssh-keygen -t ed25519 -C "email"\
ssh-copy-id picam@picam0.local\
ssh-copy-id picam@picam1.local\
ssh-copy-id picam@picam2.local\
ssh-copy-id picam@picam3.local\
ssh-copy-id picam@picam4.local\
ssh-copy-id picam@picam5.local\
ssh-copy-id tripodpi@tripodpi.local\
ssh-copy-id projectorpi@projectorpi.local\

ansible-playbook -i inventory.yaml install.yaml\
ansible-playbook -i inventory.yaml snap_all.yaml\
ansible-playbook -i inventory.yaml update_python.yaml

ssh picam@picam0.local\
python3 camera_server.py

python3 cli.py
uvicorn ui_server:app --reload
npm run dev -- --host --port 3333

ssh projectorpi@projectorpi.local
ssh tripodpi@tripodpi.local

rpicam-vid -t 0 --inline -o - | cvlc stream:///dev/stdin --sout '#rtp{sdp=rtsp://:8000/}' :demux=h264

https://calib.io/pages/camera-calibration-pattern-generator

1. charuco_build.py
2. print out that image
3. take a bunch of pictures of it using UI
4. intrinsics_calibration.py
5. stereo_calibration.py