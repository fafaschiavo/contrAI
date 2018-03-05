import random
from random import randint
import numpy as np
# import tensorflow as tf
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import time
from nes_server_v1 import NESGame
import cv2
import cPickle as pickle

def some_random_games_first(game_object):
	global screenshots_folder

	random_games_results = []
	scores_considered = []
	while len(random_games_results) < 50000:
		score = 0
		game_object.soft_reset()
		actions_history = []
		instant_history = []
		for t in range(20):
			random_number_of_actions = randint(0, 2)
			joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]

			amount_of_buttons_to_press = randint(0, 3)
			for _ in range(amount_of_buttons_to_press):
				random_button_to_press = randint(0, 5)
				joypad_list[random_button_to_press] = 1

			press_right = randint(0, 9)
			if press_right > 3:
				joypad_list[3] = 1

			game_object.send_array_joypad(joypad_list)
			p1_score = game_object.get_p1_score()
			horizontal_evolution = game_object.get_horizontal_evolution()
			score = p1_score + horizontal_evolution

			current_frame = game_object.get_last_frame_number()
			actions_history.append(joypad_list)
			if len(instant_history) == 0:
				instant_history.append(1)
			else:	
				instant_history.append(current_frame)

			time.sleep(1)
			if not game_object.is_p1_alive():
				break

		if score > 90:
			scores_considered.append(score)
			last_instant = 1
			for current_instant in instant_history[1:]:
				index = instant_history.index(current_instant)
				frames_array = game_object.get_frames_in_interval(last_instant, current_instant, black_and_white = True)
				last_instant = current_instant
				for image in frames_array:
					random_games_results.append([image, actions_history[index]])

			# print random_games_results[:10]

		# print actions_history
		# print instant_history
		print '-------------------------------------'
		print 'Total amount of frames so far: ' + str(len(random_games_results))
		print 'Current agent performance: ' + str(score)
		if len(scores_considered) > 0:
			print 'Average score considered: ' + str(mean(scores_considered)) 
			print 'Scores considered: ' + str(scores_considered)

	with open('random_examples.pickle', 'wb') as file:
		pickle.dump(random_games_results, file, protocol=pickle.HIGHEST_PROTOCOL)



print "Let's Start"

game_object = NESGame(53475, 53474)
game_object.clean_screenshots()
screenshots_folder = 'screenshots'

while not game_object.is_ready_to_listen():
	time.sleep(0.1)

some_random_games_first(game_object)

# game_object.soft_reset()
# game_object.send_array_joypad([0, 0, 0, 0, 1, 0, 0, 0])
# game_object.get_last_frame(black_and_white = True)