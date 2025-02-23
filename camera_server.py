from socket_utils import SocketUtils
import socket

def start_server(host):
    port = 65432
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    conn.sendall(data)  # Echo the received data back
    except KeyboardInterrupt:
        print("\nServer interrupted by user. Shutting down.")
    finally:
        s.close()
        print("Socket closed.")

if __name__ == '__main__':
    start_server(socket.gethostname() + ".local")