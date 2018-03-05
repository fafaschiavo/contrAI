import socket
import json
import os
import glob
from threading import Thread
import time
import keyboard
import sys
import os
import subprocess
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(sys.argv[0]), "subfolder"))

screenshots_folder = 'screenshots'
keep_socket_alive = True
data_received = False
data_to_send = False
ready_to_listen = False
p1_score = 0
p1_is_alive = False
horizontal_evolution = 0
last_frame = 0
joypad = dict(
	up=False,
	down=False,
	left=False,
	right=False,
	A=False,
	B=False,
	start=False,
	select=False,
	soft_reset=False)

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

	def send_soft_reset(self):
		global ready_to_listen
		global joypad
		ready_to_listen = False
		joypad['soft_reset'] = True
		self.UDPSock.sendto(bytes(json.dumps(joypad)), (self.IP, self.output_port))
		joypad['soft_reset'] = False

	def send_joypad(self, joypad):
		self.UDPSock.sendto(bytes(json.dumps(joypad)), (self.IP, self.output_port))

	def run(self):
		global ready_to_listen
		global keep_socket_alive
		global joypad
		global data_received
		global data_to_send
		global p1_score
		global p1_is_alive
		global horizontal_evolution
		global last_frame

		while keep_socket_alive:
			data,addr = self.UDPSock.recvfrom(1024)
			data_received = json.loads(data.strip())
			ready_to_listen = data_received['ready_to_listen']
			p1_score = data_received['p1_score']
			horizontal_evolution = data_received['horizontal_evolution']
			last_frame = data_received['last_frame']
			if data_received['p1_is_alive'] == 1:
				p1_is_alive = True
			else:
				p1_is_alive = False

			if data_to_send and ready_to_listen:
				if joypad['soft_reset']:
					ready_to_listen = False
				data_to_send = False
				print joypad
				self.UDPSock.sendto(bytes(json.dumps(joypad)), (self.IP, self.output_port))

class NESGame(object):
	global ready_to_listen
	global keep_socket_alive
	global joypad
	global data_received
	global data_to_send
	global p1_score
	global horizontal_evolution
	global joypad

	def clean_screenshots(self):
		global screenshots_folder
		files = glob.glob(screenshots_folder + '/*')
		for f in files:
			os.remove(f)

	def is_ready_to_listen(self):
		global ready_to_listen
		return ready_to_listen

	def reset_joypad_state(self):
		global joypad
		for key in joypad:
			joypad[key] = False

	def send_joypad(self, new_joypad):
		global joypad
		joypad = new_joypad
		self.socket_thread.send_joypad(new_joypad)

	def send_array_joypad(self, numpy_joypad):
		global joypad
		joypad_buttons = ['up', 'down', 'left', 'right', 'A', 'B', 'start', 'select']
		new_joypad = joypad.copy()
		new_joypad['soft_reset'] = False
		for index in range(0, len(joypad_buttons)):
			if numpy_joypad[index] == 1:
				new_joypad[joypad_buttons[index]] = True
			else:
				new_joypad[joypad_buttons[index]] = False
		joypad = new_joypad
		self.socket_thread.send_joypad(new_joypad)

	def get_current_joypad(self):
		global joypad
		return joypad

	def soft_reset(self):
		self.socket_thread.send_soft_reset()

	def is_p1_alive(self):
		global p1_is_alive
		return p1_is_alive

	def get_p1_score(self):
		global p1_score
		return p1_score

	def get_horizontal_evolution(self):
		global horizontal_evolution
		return horizontal_evolution

	def get_last_frame_number(self):
		global last_frame
		return last_frame

	def get_last_frame(self, black_and_white = False, delete_previous = True):
		global last_frame
		global screenshots_folder
		filename = screenshots_folder + '/' + str(last_frame - 1)
		print filename
		if black_and_white:
			frame = cv2.imread(filename, 0)
		else:
			frame = cv2.imread(filename)

		if delete_previous:
			files = glob.glob(screenshots_folder + '/*')
			for f in files:
				if screenshots_folder + '/' + str(last_frame) not in f:
					os.remove(f)
		return frame

	def get_frames_in_interval(self, initial_frame, last_frame, black_and_white = False):
		global screenshots_folder
		frame_array = []
		for frame in range(initial_frame, last_frame):
			filename = screenshots_folder + '/' + str(frame)
			if black_and_white:
				frame_image = cv2.imread(filename, 0)
			else:
				frame_image = cv2.imread(filename)
			frame_array.append(frame_image)

		return frame_array

	def __init__(self, input_port, output_port):
		super(NESGame, self).__init__()
		# os.system("fceux contra_rom.nes --loadlua helloworld.lua")
		self.clean_screenshots()
		self.reset_joypad_state()
		self.socket_thread = socketThread(input_port, output_port)
		self.socket_thread.daemon = True
		self.socket_thread.start()
	

# game_object = NESGame(53475, 53474)

# while True:
# 	time.sleep(3)
# 	print game_object.is_p1_alive()
	# game_object.soft_reset()

# keep_socket_alive = False
# joypad['soft_reset'] = True
# game_object.soft_reset()
# game_object.send_array_joypad([0, 0, 0, 0, 1, 0, 0, 0])
# game_object.get_last_frame(black_and_white = True)