import socket

def listen():
    check_ip = "0.0.0.0"
    check_port = 14540
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((check_ip, check_port))
        print(f"Listening on {check_ip}:{check_port}...")
        
        while True:
            data, addr = sock.recvfrom(1024)
            print(f"RECEIVED {len(data)} bytes from {addr}")
            break # Just need one packet to prove it works
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    listen()
