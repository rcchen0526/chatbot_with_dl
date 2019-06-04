import sys
import socket
import os
import json

class Client(object):
    def __init__(self, ip, port):
        try:
            socket.inet_aton(ip)
            if 0 < int(port) < 65535:
                self.ip = ip
                self.port = int(port)
            else:
                raise Exception('Port value should between 1~65535')
        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def run(self):
        print('連線建立...開始聊天')
        while True:
            msg = sys.stdin.readline()
            if msg == 'exit' + os.linesep:
                print('Bye~')
                return
            if msg != os.linesep:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((self.ip, self.port))
                        s.send(msg.encode())
                        resp = s.recv(4096).decode()
                        resp = json.loads(resp)
                        print(resp['message'])
                except Exception as e:
                    print(e, file=sys.stderr)

def launch_client(ip, port):
    c = Client(ip, port)
    c.run()

if __name__ == '__main__':
    launch_client('127.0.0.1', 39391)
