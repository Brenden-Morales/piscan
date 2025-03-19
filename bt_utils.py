from time import sleep

import bluetooth
from bluetooth.btcommon import Protocols
import struct

class BTUtils:
    def __init__(self):
        self.bt_name = "ESP32_Stepper"
        self.port = 1

        print("Performing bluetooth inquiry...")
        nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True,flush_cache=True, lookup_class=False)
        print("Found {} bluetooth devices".format(len(nearby_devices)))
        for addr, name in nearby_devices:
            try:
                if name == self.bt_name:
                    self.mac_addr = addr
                    print('Got MAC address of bluetooth stepper')
            except UnicodeEncodeError:
                print("   {} - {}".format(addr, name.encode("utf-8", "replace")))
        if self.mac_addr:
            print("connecting to {} with MAC {} on port {}".format(self.bt_name, self.mac_addr, self.port))
            self.sock = bluetooth.BluetoothSocket(Protocols.RFCOMM)
            self.sock.connect((self.mac_addr, self.port))
            print("connected")

    def step(self, num_steps: int):
        self.sock.send(struct.pack("i", num_steps))

    def close(self):
        self.sock.close()