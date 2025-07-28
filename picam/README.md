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