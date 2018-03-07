import gym
import random
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
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
		self.output = fully_connected(self.second_fully_connected, output_size, activation='softmax')
		self.network = regression(self.output, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
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
		with tf.Graph().as_default():
			individual = genetic_model(input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)
			individual.initialize_random_weights_and_biases()
			individual.model.fit({'input': X}, {'targets': Y}, n_epoch = n_epoch, snapshot_step = snapshot_step, snapshot_epoch = snapshot_epoch, show_metric = False)
			current_generation_individuals.append(individual)

	return current_generation_individuals

def initial_population():
	print 'Generating initial dandom data for training...'
	training_data = []
	scores = []
	accepted_scores = []
	while len(training_data) < 200:
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(200):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)
			
			if len(prev_observation) > 0 :
				game_memory.append([prev_observation, action])
			prev_observation = observation
			score+=reward
			if done: break

		if score >= 40:
			accepted_scores.append(score)
			for data in game_memory:
				action_output = np.zeros(output_size)
				action_output[data[1]] = 1
				training_data.append([data[0], action_output])
				
		env.reset()
		scores.append(score)
	
	training_data_save = np.array(training_data)
	
	print '---------------------------------------------------'
	print 'Initial random data for training:'
	print 'Average accepted score:', mean(accepted_scores)
	print 'Median score for accepted scores:',median(accepted_scores)
	print 'Amount of random frames to use as initial data: ', len(training_data)
	print '---------------------------------------------------'
	return training_data

print '============'
print "Let's Start"
print '============'

generation_report = 'cart_pole_v2.txt'

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 195

input_size = 4
output_size = 2
first_fully_connected_size = 20
second_fully_connected_size = 20
activation = 'relu'
individuals_per_generation = 10
n_epoch = 5

snapshot_step = None
snapshot_epoch = False

# Initial generation
initial_training_data = initial_population()

X = [np.array([data[0]]).reshape(input_size, 1) for data in initial_training_data]
Y = [data[1] for data in initial_training_data]
current_generation_individuals = train_new_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step, snapshot_epoch)
# print initial_training_data

current_generation = 0
best_performance = 0
mean_score = 0
while mean_score < goal_steps:
	print '--------------------------'
	print 'Starting Generation: ' + str(current_generation)
	print '--------------------------'

	generation_scores = []
	generation_game_memory = []
	for individual in current_generation_individuals:
		individual_game_memory = []
		individual_performance = 0
		env.reset()

		for t in range(500):
			env.render()

			if t == 0:
				action = env.action_space.sample()
			else:
				action_output = individual.model.predict(observation.reshape(-1, 4, 1))
				action = np.argmax(action_output[0])
				action_output = np.zeros(output_size)
				action_output[action] = 1
				individual_game_memory.append([observation, action_output])

			observation, reward, done, info = env.step(action)

			individual_performance = individual_performance + 1
			if done:
				break
		print 'Individual Performance: ' + str(individual_performance)
		generation_scores.append(individual_performance)
		generation_game_memory.append(individual_game_memory)

	best_performance = max(generation_scores)
	best_individual = generation_scores.index(best_performance)

	mean_score = mean(generation_scores)

	print '--------------------------'
	print 'Best score in this generation: ' + str(best_performance)
	print 'Best individual: ' + str(best_individual)
	print 'Mean performance in the generation: ' + str(mean_score)
	print 'Median performance in the generation: ' + str(median(generation_scores))

	if os.path.exists(generation_report):
		append_write = 'a'
	else:
		append_write = 'w'
	highscore = open(generation_report, append_write)
	highscore.write("Generation_" + str(current_generation) + ';' + str(best_performance) + ';' + str(median(generation_scores)) + ';' + str(len(X)) + '\n')
	highscore.close()

	current_generation = current_generation + 1

	if best_performance >= goal_steps:
		index = 0
		for score in generation_scores:
			if score >= goal_steps:
				new_X = [np.array(data[0]).reshape(input_size, 1) for data in generation_game_memory[index]]
				X = X + new_X
				new_Y = [data[1] for data in generation_game_memory[index]]
				Y = Y + new_Y
			index = index + 1
	else:
		new_X = [np.array(data[0]).reshape(input_size, 1) for data in generation_game_memory[best_individual]]
		X = X + new_X
		new_Y = [data[1] for data in generation_game_memory[best_individual]]
		Y = Y + new_Y

	current_generation_individuals = train_new_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation, X, Y, n_epoch, snapshot_step, snapshot_epoch)





generation_scores = []
generation_game_memory = []
for individual in current_generation_individuals:
	individual_game_memory = []
	individual_performance = 0
	env.reset()

	for t in range(500):
		env.render()

		if t == 0:
			action = env.action_space.sample()
		else:
			action_output = individual.model.predict(observation.reshape(-1, 4, 1))
			action = np.argmax(action_output[0])
			action_output = np.zeros(output_size)
			action_output[action] = 1
			individual_game_memory.append([observation, action_output])

		observation, reward, done, info = env.step(action)

		individual_performance = individual_performance + 1
		if done:
			break
	print 'Individual Performance: ' + str(individual_performance)
	generation_scores.append(individual_performance)
	generation_game_memory.append(individual_game_memory)

best_performance = max(generation_scores)
best_individual = generation_scores.index(best_performance)

mean_score = mean(generation_scores)

print '--------------------------'
print 'Best score in the last generation: ' + str(best_performance)
print 'Mean performance in the generation: ' + str(mean_score)
print 'Median performance in the last generation: ' + str(median(generation_scores))