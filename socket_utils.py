import struct
import socket

class SocketUtils:
    @staticmethod
    def send_message(sock, message):
        # Encode message and get its length
        encoded_msg = message
        msg_length = len(encoded_msg)
        # Pack the length into 4 bytes using network byte order
        sock.sendall(struct.pack('!I', msg_length))
        sock.sendall(encoded_msg)

    @staticmethod
    def receive_message(sock):
        # First, read 4 bytes to get the message length
        raw_length = sock.recv(4)
        if not raw_length:
            return None
        msg_length = struct.unpack('!I', raw_length)[0]
        # Now receive the actual message
        data = bytearray()
        while len(data) < msg_length:
            packet = sock.recv(4096)
            if not packet:
                break
            data.extend(packet)
        return data