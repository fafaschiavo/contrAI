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

def get_some_random_action():
	random_number_of_actions = randint(0, 2)
	joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]

	amount_of_buttons_to_press = randint(0, 3)
	for _ in range(amount_of_buttons_to_press):
		random_button_to_press = randint(0, 5)
		joypad_list[random_button_to_press] = 1

	press_right = randint(0, 9)
	if press_right > 6:
		joypad_list[3] = 1
	return joypad_list

def some_random_games_first(game_object):
	global screenshots_folder

	random_games_results = []
	scores_considered = []
	while len(random_games_results) < 50000:
		score = 0
		game_object.soft_reset()
		game_memory = []
		joypad_list = get_some_random_action()
		for t in range(600):

			if t%10 == 0:
				joypad_list = get_some_random_action()

			game_object.send_array_joypad(joypad_list)
			p1_score = game_object.get_p1_score()
			horizontal_evolution = game_object.get_horizontal_evolution()
			score = (100*p1_score) + horizontal_evolution

			current_features = game_object.get_array_features()
			game_memory.append([current_features, joypad_list])
			time.sleep(0.01)
			if not game_object.is_p1_alive():
				break

		if score > 90:
			game_object.soft_reset()
			time.sleep(0.5)
			scores_considered.append(score)
			random_games_results = random_games_results + game_memory
			# print random_games_results[:10]

		# print actions_history
		# print instant_history
		print '-------------------------------------'
		print 'Total amount of frames so far: ' + str(len(random_games_results))
		print 'Current agent performance: ' + str(score)
		if len(scores_considered) > 0:
			print 'Average score considered: ' + str(mean(scores_considered)) 
			print 'Scores considered: ' + str(scores_considered)

	with open('v2_random_examples_10k_0_02_high_score_value_activity.pickle', 'wb') as file:
		pickle.dump(random_games_results, file, protocol=pickle.HIGHEST_PROTOCOL)



print "Let's Start"

game_object = NESGame(53475, 53474)
game_object.clean_screenshots()
screenshots_folder = 'screenshots'

while not game_object.is_ready_to_listen():
	time.sleep(0.1)

game_object.soft_reset()
some_random_games_first(game_object)

# game_object.soft_reset()
# game_object.send_array_joypad([0, 0, 0, 0, 1, 0, 0, 0])
# game_object.get_last_frame(black_and_white = True)