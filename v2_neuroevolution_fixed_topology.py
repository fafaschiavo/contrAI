import random
from random import randint
from random import shuffle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import time
from nes_server_v1 import NESGame
import cv2
import cPickle as pickle
import os

class genetic_model(object):
	input_size = 0
	output_size = 0
	first_fully_connected_size = 0
	second_fully_connected_size = 0
	activation = 0

	def __init__(self, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation):
		super(genetic_model, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.first_fully_connected_size = first_fully_connected_size
		self.second_fully_connected_size = second_fully_connected_size
		self.activation = activation

		self.input_layer = input_data(shape=[None, input_size, 1], name='input')
		self.first_fully_connected = fully_connected(self.input_layer, first_fully_connected_size, activation=activation)
		self.second_fully_connected = fully_connected(self.first_fully_connected, second_fully_connected_size, activation=activation)
		self.output = fully_connected(self.second_fully_connected, output_size, activation='linear')
		self.network = regression(self.output, optimizer='sgd', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
		self.model = tflearn.DNN(self.network, tensorboard_dir='log')
		
	def initialize_random_weights_and_biases(self):
		self.model.set_weights(self.first_fully_connected.W, np.random.rand(self.input_size, self.first_fully_connected_size))
		self.model.set_weights(self.first_fully_connected.b, np.random.rand(self.first_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.W, np.random.rand(self.first_fully_connected_size, self.second_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.b, np.random.rand(self.second_fully_connected_size))
		self.model.set_weights(self.output.W, np.random.rand(self.second_fully_connected_size, self.output_size))
		self.model.set_weights(self.output.b, np.random.rand(self.output_size))

def get_empty_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation):
	current_generation_individuals = []
	for individual_index in xrange(0, individuals_per_generation):
		with tf.Graph().as_default():
			individual = genetic_model(input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			current_generation_individuals.append(individual)
	return current_generation_individuals

def train_new_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step, snapshot_epoch):
	print 'Training new generation....'
	current_generation_individuals = []
	for individual_index in xrange(0, individuals_per_generation):
		print '-------------------------------------'
		print 'Now training inidividual ' + str(individual_index)
		print '-------------------------------------'
		with tf.Graph().as_default():
			individual = genetic_model(input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			individual.model.fit({'input': X}, {'targets': Y}, n_epoch = n_epoch, snapshot_step = snapshot_step, snapshot_epoch = snapshot_epoch, show_metric = False)
			current_generation_individuals.append(individual)

	return current_generation_individuals





print "Let's Start"

pickle_to_load = 'v2_random_examples_10k_0_02_high_score_value_activity.pickle'
models_folder = 'models/'
generation_report = 'v2_generation_report.txt'

input_size = 184
output_size = 8
first_fully_connected_size = 10
second_fully_connected_size = 10
activation = 'relu'
mutation_rate = 0.2
individuals_per_generation = 2
n_epoch = 20
initial_data = 10000
goal_score = 600

snapshot_step = None
snapshot_epoch = True

with open(pickle_to_load, 'rb') as file:
	initial_training_data = pickle.load(file)

np.random.shuffle(initial_training_data)
initial_training_data = initial_training_data[:initial_data]

# game_object = NESGame(53475, 53474)
game_object = NESGame(53476, 53477)
game_object.clean_screenshots()
screenshots_folder = 'screenshots'

X = [data[0].reshape(input_size, 1) for data in initial_training_data]
Y = [data[1] for data in initial_training_data]

current_generation_individuals = train_new_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step, snapshot_epoch)

while not game_object.is_ready_to_listen():
	time.sleep(0.1)

current_generation = 0
best_performance = 0
mean_score = 0
while mean_score < goal_score:
	print '-------------------------------------'
	print 'Starting Generation: ' + str(current_generation)
	print '-------------------------------------'

	generation_scores = []
	generation_game_memory = []
	score_history = []
	game_object.soft_reset()
	for individual in current_generation_individuals:

		# reset game and wait for it to finish reseting
		game_object.soft_reset()
		time.sleep(2)

		actions_history = []
		game_memory = []
		score = 0
		individual_performance = 0
		old_age_verification = 0
		last_old_age_verification = 0
		repetitiviness_rate = 0
		repetitiviness_history = []
		while True:
			current_frame_number = game_object.get_last_frame_number()

			# Check if agent is only repeating the same command
			repetitiviness_rate = len(set(repetitiviness_history))
			if len(repetitiviness_history) == 10000 and repetitiviness_rate == 1:
				print 'Current argmax set: ' + str(set(repetitiviness_history))
				print 'Dying of repetitiviness....'
				break

			# Check if agent should die of OLD AGE
			if current_frame_number % 400 == 0:
				if score == old_age_verification and last_old_age_verification != current_frame_number:
					print 'Current frame: ' + str(game_object.get_last_frame_number())
					print 'Current Score: ' + str(score)
					print 'Old age Score: ' + str(old_age_verification)
					print 'Dying of old age....'
					break
				else:
					old_age_verification = score
					last_old_age_verification = current_frame_number

			current_feature_set = game_object.get_array_features()
			if current_feature_set is not None:
				# Get current status, analyse agent output and sent command
				current_feature_set = current_feature_set.reshape(-1, input_size, 1)
				prediction = individual.model.predict(current_feature_set)
				sorted_prediction = prediction.argsort()[0][-2:]
				joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]
				for index in sorted_prediction:
					joypad_list[index] = 1
				game_object.send_array_joypad(joypad_list)

				# Save decision for posteriority
				actions_history.append(joypad_list)
				game_memory.append(current_feature_set)
				repetitiviness_history.append(np.argmax(joypad_list))

				# Calculate score
				p1_score = game_object.get_p1_score()
				horizontal_evolution = game_object.get_horizontal_evolution()
				score = ((100*p1_score) + horizontal_evolution)*repetitiviness_rate

				# Check if player is dead
				if not game_object.is_p1_alive():
					break

				# time.sleep(0.1)
			else:
				pass

		score_history.append(score)
		print 'Current agent game score: ' + str(p1_score) 
		print 'Current agent horizontal evolution: ' + str(horizontal_evolution)
		print 'Current agent repetitiviness rate: ' + str(repetitiviness_rate)
		print 'Current agent performance: ' + str(score)
		print '-------------------------------------'

		joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]
		game_object.send_array_joypad(joypad_list)
		game_object.soft_reset()
		time.sleep(3)