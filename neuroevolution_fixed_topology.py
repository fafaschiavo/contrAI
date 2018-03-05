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

class genetic_model(object):
	input_size = 0
	output_size = 0
	first_fully_connected_size = 0
	second_fully_connected_size = 0
	activation = 0

	def __init__(self, input_size_width, input_size_heigth, output_size, first_fully_connected_size, second_fully_connected_size, activation):
		super(genetic_model, self).__init__()

		self.input_size_width = input_size_width
		self.input_size_heigth = input_size_heigth
		self.output_size = output_size
		self.first_fully_connected_size = first_fully_connected_size
		self.second_fully_connected_size = second_fully_connected_size
		self.activation = activation

		self.input_layer = input_data(shape=[None, input_size_width, input_size_heigth, 1], name='input')
		self.first_fully_connected = fully_connected(self.input_layer, first_fully_connected_size, activation=activation)
		self.second_fully_connected = fully_connected(self.first_fully_connected, second_fully_connected_size, activation=activation)
		self.output = fully_connected(self.second_fully_connected, output_size, activation='softplus')
		self.network = regression(self.output, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
		self.model = tflearn.DNN(self.network, tensorboard_dir='log')
		
	def initialize_random_weights_and_biases(self):
		self.model.set_weights(self.first_fully_connected.W, np.random.rand(self.input_size_width*self.input_size_heigth, self.first_fully_connected_size))
		self.model.set_weights(self.first_fully_connected.b, np.random.rand(self.first_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.W, np.random.rand(self.first_fully_connected_size, self.second_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.b, np.random.rand(self.second_fully_connected_size))
		self.model.set_weights(self.output.W, np.random.rand(self.second_fully_connected_size, self.output_size))
		self.model.set_weights(self.output.b, np.random.rand(self.output_size))

def get_empty_generation(individuals_per_generation, input_size_width, input_size_heigth, output_size, first_fully_connected_size, second_fully_connected_size, activation):
	current_generation_individuals = []
	for individual_index in xrange(0, individuals_per_generation):
		with tf.Graph().as_default():
			individual = genetic_model(input_size_width, input_size_heigth, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			current_generation_individuals.append(individual)
	return current_generation_individuals

def train_new_generation(individuals_per_generation, input_size_width, input_size_heigth, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step, snapshot_epoch):
	print 'Training new generation....'
	current_generation_individuals = []
	for individual_index in xrange(0, individuals_per_generation):
		with tf.Graph().as_default():
			individual = genetic_model(input_size_width, input_size_heigth, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			individual.model.fit({'input': X}, {'targets': Y}, n_epoch = n_epoch, snapshot_step = snapshot_step, snapshot_epoch = snapshot_epoch, show_metric = False)
			current_generation_individuals.append(individual)

	return current_generation_individuals

print "Let's Start"

pickle_to_load = 'random_examples_50k_bigger90.pickle'

input_size_width = 224
input_size_heigth = 256
output_size = 8
first_fully_connected_size = 50
second_fully_connected_size = 50
activation = 'relu'
individuals_per_generation = 3
n_epoch = 5
goal_score = 600

snapshot_step = None
snapshot_epoch = True

with open(pickle_to_load, 'rb') as file:
	initial_training_data = pickle.load(file)

np.random.shuffle(initial_training_data)
initial_training_data = initial_training_data[:10000]

game_object = NESGame(53475, 53474)
game_object.clean_screenshots()
screenshots_folder = 'screenshots'

X = [data[0].reshape(input_size_width, input_size_heigth, 1) for data in initial_training_data]
Y = [data[1] for data in initial_training_data]
current_generation_individuals = train_new_generation(individuals_per_generation, input_size_width, input_size_heigth, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step, snapshot_epoch)

while not game_object.is_ready_to_listen():
	time.sleep(0.1)

current_generation = 0
best_performance = 0
mean_score = 0
while mean_score < goal_score:
	print '--------------------------'
	print 'Starting Generation: ' + str(current_generation)
	print '--------------------------'

	generation_scores = []
	generation_game_memory = []
	for individual in current_generation_individuals:
		individual_performance = 0
		game_object.soft_reset()
		time.sleep(0.1)
		while game_object.is_p1_alive():
			current_frame = game_object.get_last_frame(black_and_white = True, delete_previous = False)
			if current_frame is not None:
				current_frame = current_frame.reshape(-1, input_size_width, input_size_heigth, 1)
				prediction = individual.model.predict(current_frame)
				sorted_prediction = prediction.argsort()[0][-2:]
				joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]
				for index in sorted_prediction:
					joypad_list[index] = 1

				game_object.send_array_joypad(joypad_list)
			else:
				pass