import socket
from socket_utils import SocketUtils

def start_client(host):
    port = 65432        # Must be the same as the server port

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        # Send a message to the server
        message = "Hello, server!"
        SocketUtils.send_message(sock, message.encode())
        print("Sent to camera server {}:{}: {}".format(host, port, message))
        # Receive response from the server
        data = SocketUtils.receive_message(sock)
        print("Received from server {}:{}: {}".format(host, port, data.decode()))

if __name__ == '__main__':
    start_client("picam0.local")