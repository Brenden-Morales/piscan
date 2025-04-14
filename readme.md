ssh-keygen -t ed25519 -C "email"\
ssh-copy-id picam@picam0.local\
ssh-copy-id picam@picam1.local\
ssh-copy-id picam@picam2.local\
ssh-copy-id picam@picam3.local\
ssh-copy-id picam@picam4.local\
ssh-copy-id picam@picam5.local\

ansible-playbook -i inventory.yaml install.yaml\
ansible-playbook -i inventory.yaml snap_all.yaml\
ansible-playbook -i inventory.yaml update_python.yaml

ssh picam@picam0.local\
python3 camera_server.py

python3 cli.py
uvicorn ui_server:app --reload
npm run dev -- --host --port 3333

rpicam-vid -t 0 --inline -o - | cvlc stream:///dev/stdin --sout '#rtp{sdp=rtsp://:8000/}' :demux=h264