ansible-playbook -i inventory.yaml install.yaml
ansible-playbook -i inventory.yaml snap_all.yaml

rpicam-vid -t 0 --inline -o - | cvlc stream:///dev/stdin --sout '#rtp{sdp=rtsp://:8000/}' :demux=h264