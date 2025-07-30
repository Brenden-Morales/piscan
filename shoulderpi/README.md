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

## ansible
Deploying and managing the pi5 in this project is handled by [ansible](https://docs.ansible.com/ansible/latest/getting_started/index.html)

```shell
ansible-playbook -i shoulderpi/inventory.yaml shoulderpi/ping.yaml
```