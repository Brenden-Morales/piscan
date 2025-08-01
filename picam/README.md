# picam
This is for the code and infrastructure that will run multiple raspberry pi zero 2w modules with cameras attached

## hardware
- [pi zero 2w](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/)
- [pi camera module 3](https://www.raspberrypi.com/products/camera-module-3/)



## setup

1. Install [raspberry pi imager](https://www.raspberrypi.com/software/)
2. Flash latest image to each zero 2w
   3. Start with `picam0`, `picam1`, `picam2` etc
3. Set username to `picam` and password to whatever works for you
4. make sure to flash with wifi (if using wifi to connect)
5. Setup SSH creds
```shell
ssh-keygen -t ed25519 -C "email"
ssh-copy-id picam@picamX.local
ssh picam@picamX.local
```
6. Setup [VNC](https://www.raspberrypi.com/documentation/computers/remote-access.html#vnc) if needed

## ansible
Deploying and managing the pi zero 2Ws in this project is handled by [ansible](https://docs.ansible.com/ansible/latest/getting_started/index.html)

```shell
ansible-playbook -i picam/ansible/inventory.yaml picam/ansible/playbook.yaml
```

This will put all the pi zero 2w's into ethernet gadget mode. Plug them into a powered hub via the `usb` connection 
on the bottom, make sure the hub is connected to the pi5 

The connection parameters for the pi to communicate back to the host will be in `/etc/picam_config.json` and will look like:
```json
{
   "name": "picam0.local", 
   "usb_dev_mac": "12:22:33:44:55:00", 
   "usb_host_mac": "16:22:33:44:55:00", 
   "usb_static_ip": "192.168.7.11", 
   "usb_peer_ip": "192.168.7.10", 
   "listen_port": 9000
}
```