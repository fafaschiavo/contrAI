import socket
import json
import os
import glob

IP = '127.0.0.1'

joypad = dict(
	up=False,
	down=False,
	left=False,
	right=False,
	A=False,
	B=False,
	start=False,
	select=False)

UDPSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

listen_addr = ("",53475)
UDPSock.bind(listen_addr)

files = glob.glob('screenshots/*')
for f in files:
    os.remove(f)

while True:
	data,addr = UDPSock.recvfrom(1024)
	received_data = json.loads(data.strip())
	print received_data['ready_to_listen']
	print received_data['p1_score']
	print received_data['horizontal_evolution']
	print '--------------'

	data = dict(joypad=joypad, time=150)
	UDPSock.sendto(bytes(json.dumps(data)), (IP, 53474))
	# print data