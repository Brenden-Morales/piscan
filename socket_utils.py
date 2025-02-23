import struct
import socket
import math

class SocketUtils:
    TAKE_SNAPSHOT = "SNAPSHOT"
    chunk_size = 1024 * 4

    @staticmethod
    def send_message(sock, message):
        # Encode message and get its length
        msg_length = len(message)
        print("message length: {}".format(msg_length))
        # Pack the length into 4 bytes using network byte order
        sock.sendall(struct.pack('!I', msg_length))
        # Then send the image data in manageable chunks
        bytes_sent = 0
        iterations = 0
        print("total chunks: {}".format(math.ceil(msg_length / SocketUtils.chunk_size)))
        while bytes_sent < msg_length:
            iterations += 1
            print("sending chunk: {}".format(iterations))
            chunk = message[bytes_sent:bytes_sent+SocketUtils.chunk_size]
            sock.sendall(chunk)
            bytes_sent += len(chunk)

    @staticmethod
    def receive_message(sock):
        # First, read 4 bytes to get the message length
        raw_length = sock.recv(4)
        if not raw_length:
            return None
        msg_length = struct.unpack('!I', raw_length)[0]
        print("receive message of length: {}".format(msg_length))
        # Now receive the actual message
        data = bytearray()
        chunks_received = 0
        while len(data) < msg_length:
            packet = sock.recv(SocketUtils.chunk_size)
            chunks_received += 1
            print("received chunk: {}".format(chunks_received))
            if not packet:
                break
            data.extend(packet)
        return data