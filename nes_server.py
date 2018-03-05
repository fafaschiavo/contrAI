import socket
import json
import os
import glob
from threading import Thread
import time
import sys
import os
import subprocess
import cv2
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), "subfolder"))

screenshots_folder = 'screenshots'
keep_socket_alive = True
data_received = ''
data_to_send = False

class socketThread(Thread):
	IP = '127.0.0.1'
	input_port = 53475
	output_port = 53474

	def __init__(self, input_port, output_port):
		Thread.__init__(self)
		self.input_port = input_port
		self.output_port = output_port
		self.UDPSock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.listen_addr = (self.IP,input_port)
		self.UDPSock.bind(self.listen_addr)

	def run(self):
		global keep_socket_alive
		global data_received
		global data_to_send

		while True:
			if data_to_send:
				print 'flag 3'
				self.UDPSock.sendto(bytes(json.dumps(data_to_send)), (self.IP, self.output_port))
				data_to_send = False


socket_thread = socketThread(53475, 53474)
socket_thread.daemon = True
socket_thread.start()

while True:
	time.sleep(0.5)
	print 'flag 1'
	data_to_send = {
		'command': 'frameadvance',
	}