# ULTIMATE COMBINED ANSIBLE

```shell
source .venv/bin/activate
# provision devices with dependencies etc
ansible-playbook -i ansible/inventory.yaml ansible/provision.yaml
# application deployment
ansible-playbook -i ansible/inventory.yaml ansible/deploy_shoulder.yaml
ansible-playbook -i ansible/inventory.yaml ansible/deploy_cameras.yaml
```

---

# picams
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

---

# shoulderpi
This is for the code and infrastructure that will need to run on the raspberry pi 5 that is on the shoulder rig

## Hardware
- [raspberry pi 5 (8GB)](https://www.raspberrypi.com/products/raspberry-pi-5/)
- [pi 5 active cooler](https://www.raspberrypi.com/products/active-cooler/)
- [5.5" 2K capacitive LCD](https://www.waveshare.com/5.5inch-1440x2560-lcd.htm)
- [Raspberry PI AI Hat (13 TOPS)](https://www.raspberrypi.com/products/ai-hat/?variant=ai-hat-plus-13)
- [Shoulder Rig Kit](https://www.smallrig.com/Shoulder-Rig-Kit-Classic-Version-4480.html)
- [USB Hub](https://www.amazon.com/dp/B00VDVCQ84)
- [micro USB cables](https://www.amazon.com/dp/B095JZSHXQ)
- [Makita Battery Terminal](https://www.amazon.com/dp/B0DPSPLWFB)
- [low voltage module](https://www.amazon.com/dp/B08H14XTZ8)
- [ideal diodes](https://www.amazon.com/dp/B0DDJFBF3B)
- [DC -> DC converter](https://www.digikey.com/en/products/detail/mean-well-usa-inc/RSD-60G-12/7706258)


## setup

1. Install [raspberry pi imager](https://www.raspberrypi.com/software/)
2. Flash latest image to pi 5
3. Set username to `shoulderpi` and password to whatever works for you
4. make sure to flash with wifi (if using wifi to connect)
5. Setup SSH creds
```shell
ssh-keygen -t ed25519 -C "email"
ssh-copy-id shoulderpi@shoulderpi.local
ssh shoulderpi@shoulderpi.local
```
6. Setup [VNC](https://www.raspberrypi.com/documentation/computers/remote-access.html#vnc) if needed
7. Install [ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html#installing-and-upgrading-ansible) to manage deployments to the pi 5

## ansible
Deploying and managing the pi5 in this project is handled by [ansible](https://docs.ansible.com/ansible/latest/getting_started/index.html)

That should set up all the links to the picams for networking AND create a file at:
```shell
/etc/picam_links.json
```

That will have the config so that other scripts can use it. e.x.:

```json
[
    {
        "host_ip": "192.168.7.10",
        "mac": "16:22:33:44:55:00",
        "name": "picam0",
        "peer_ip": "192.168.7.11"
    }, ...
]
```