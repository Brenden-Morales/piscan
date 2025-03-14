ssh-keygen -t ed25519 -C "email"\
ssh-copy-id picam@picam0.local\
ssh-copy-id picam@picam1.local\
ssh-copy-id picam@picam2.local\
ssh-copy-id picam@picam3.local\

ansible-playbook -i inventory.yaml install.yaml\
ansible-playbook -i inventory.yaml snap_all.yaml\
ansible-playbook -i inventory.yaml update_python.yaml

python camera_controller.py

ssh picam@picam0.local\
python3 camera_server.py

rpicam-vid -t 0 --inline -o - | cvlc stream:///dev/stdin --sout '#rtp{sdp=rtsp://:8000/}' :demux=h264