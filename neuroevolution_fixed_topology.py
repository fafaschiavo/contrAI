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
		
	def get_model(self):
		return self.model

	def initialize_random_weights_and_biases(self):
		self.model.set_weights(self.first_fully_connected.W, np.random.rand(self.input_size_width*self.input_size_heigth, self.first_fully_connected_size))
		self.model.set_weights(self.first_fully_connected.b, np.random.rand(self.first_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.W, np.random.rand(self.first_fully_connected_size, self.second_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.b, np.random.rand(self.second_fully_connected_size))
		self.model.set_weights(self.output.W, np.random.rand(self.second_fully_connected_size, self.output_size))
		self.model.set_weights(self.output.b, np.random.rand(self.output_size))

	def generate_mutations(self, mutation_rate):
		genes_to_mutate = [
			self.first_fully_connected.W,
			self.first_fully_connected.b,
			self.second_fully_connected.W,
			self.second_fully_connected.b,
			self.output.W,
			self.output.b,
		]

		for weight_portion in genes_to_mutate:
			gene = self.model.get_weights(weight_portion)

			if len(gene.shape) == 1:
				for i in range(0, gene.shape[0]):
					will_mutate = random.uniform(0, 1)
					if will_mutate < mutation_rate:
						new_weight = random.uniform(0, 1)
						gene[i] = new_weight
				self.model.set_weights(weight_portion, gene)

			if len(gene.shape) == 2:
				for i in range(0, gene.shape[0]):
					for j in range(0, gene.shape[1]):
						will_mutate = random.uniform(0, 1)
						if will_mutate < mutation_rate:
							new_weight = random.uniform(0, 1)
							gene[i][j] = new_weight
				self.model.set_weights(weight_portion, gene)

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
models_folder = 'models/'
generation_report = 'generation_report.txt'

input_size_width = 224
input_size_heigth = 256
output_size = 8
first_fully_connected_size = 50
second_fully_connected_size = 50
activation = 'relu'
mutation_rate = 0.2
individuals_per_generation = 10
n_epoch = 6
initial_data = 10000
goal_score = 600

snapshot_step = None
snapshot_epoch = True

with open(pickle_to_load, 'rb') as file:
	initial_training_data = pickle.load(file)

np.random.shuffle(initial_training_data)
initial_training_data = initial_training_data[:initial_data]

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
	print '-------------------------------------'
	print 'Starting Generation: ' + str(current_generation)
	print '-------------------------------------'

	generation_scores = []
	generation_game_memory = []
	score_history = []
	game_object.soft_reset()
	for individual in current_generation_individuals:
		individual_performance = 0
		game_object.soft_reset()
		time.sleep(2)
		actions_history = []
		frame_history = []
		score = 0
		old_age_verification = 0
		last_old_age_verification = 0
		while True:
			current_frame_number = game_object.get_last_frame_number()
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
			current_frame = game_object.get_last_frame(black_and_white = True, delete_previous = False)
			if current_frame is not None:
				current_frame = current_frame.reshape(-1, input_size_width, input_size_heigth, 1)
				prediction = individual.model.predict(current_frame)
				sorted_prediction = prediction.argsort()[0][-2:]
				joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]
				for index in sorted_prediction:
					joypad_list[index] = 1
				game_object.send_array_joypad(joypad_list)

				actions_history.append(joypad_list)
				frame_history.append(current_frame)

				p1_score = game_object.get_p1_score()
				horizontal_evolution = game_object.get_horizontal_evolution()
				score = p1_score + horizontal_evolution
				if not game_object.is_p1_alive():
					break
			else:
				pass

		score_history.append(score)
		print 'Current agent performance: ' + str(score)
		print '-------------------------------------'

		joypad_list = [0, 0, 0, 0, 0, 0, 0, 0]
		game_object.send_array_joypad(joypad_list)
		game_object.soft_reset()
		time.sleep(3)

	index = 0
	best_score = np.argmax(score_history)
	for score in score_history:
		if index != best_score:
			print 'Mutating now....'
			current_generation_individuals[index].generate_mutations(mutation_rate)
		else:
			if score > 2000:
				try:
					name_to_save = 'model_generation_' + str(current_generation) + '_epochs_' + str(n_epoch) + '_initialdata_' + str(initial_data)+'.tflearn'
					current_generation_individuals[index].model.save(models_folder + name_to_save)
				except Exception as e:
					print e
		index = index + 1

	if os.path.exists(generation_report):
		append_write = 'a'
	else:
		append_write = 'w'
	highscore = open(generation_report, append_write)
	highscore.write("Generation_" + str(current_generation) + ';' + str(score_history[best_score]) + '\n')
	highscore.close()

	current_generation = current_generation + 1