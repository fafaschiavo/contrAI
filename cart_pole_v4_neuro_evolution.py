import gym
import random
from random import randint
from random import shuffle
from random import choice
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import time
import os
import heapq

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
		self.network = regression(self.output, optimizer='sgd', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
		self.model = tflearn.DNN(self.network, tensorboard_dir='log')
		
	def initialize_random_weights_and_biases(self):
		self.model.set_weights(self.first_fully_connected.W, np.random.rand(self.input_size, self.first_fully_connected_size))
		self.model.set_weights(self.first_fully_connected.b, np.random.rand(self.first_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.W, np.random.rand(self.first_fully_connected_size, self.second_fully_connected_size))
		self.model.set_weights(self.second_fully_connected.b, np.random.rand(self.second_fully_connected_size))
		self.model.set_weights(self.output.W, np.random.rand(self.second_fully_connected_size, self.output_size))
		self.model.set_weights(self.output.b, np.random.rand(self.output_size))

	def clone_individual(self, second_individual):
		self.model.set_weights(self.first_fully_connected.W, second_individual.model.get_weights(second_individual.first_fully_connected.W))
		self.model.set_weights(self.first_fully_connected.b, second_individual.model.get_weights(second_individual.first_fully_connected.b))
		self.model.set_weights(self.second_fully_connected.W, second_individual.model.get_weights(second_individual.second_fully_connected.W))
		self.model.set_weights(self.second_fully_connected.b, second_individual.model.get_weights(second_individual.second_fully_connected.b))
		self.model.set_weights(self.output.W, second_individual.model.get_weights(second_individual.output.W))
		self.model.set_weights(self.output.b, second_individual.model.get_weights(second_individual.output.b))

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

	def generate_crossover(self, second_individual):
		first_gene = self.model.get_weights(self.first_fully_connected.W)
		second_gene = second_individual.model.get_weights(second_individual.first_fully_connected.W)
		new_gene = crossover_genes(first_gene, second_gene)
		self.model.set_weights(self.first_fully_connected.W, new_gene)

def crossover_genes(first_gene, second_gene):
	new_gene = np.zeros(first_gene.shape)
	if len(new_gene.shape) == 2:
		for i in range(0,new_gene.shape[0]):
			for j in range(0,new_gene.shape[1]):
				if randint(0, 1) == 0:
					new_gene[i][j] = first_gene[i][j]
				else:
					new_gene[i][j] = second_gene[i][j]

	if len(new_gene.shape) == 1:
		for i in range(0,new_gene.shape[0]):
			if randint(0, 1) == 0:
				new_gene[i] = first_gene[i]
			else:
				new_gene[i] = second_gene[i]

	return new_gene

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

print '============'
print "Let's Start"
print '============'

generation_report = 'cart_pole_v4_v10.txt'

env = gym.make("CartPole-v0")
env.reset()
goal_steps = 195

input_size = 4
output_size = 2
first_fully_connected_size = 30
second_fully_connected_size = 30
activation = 'relu'
n_epoch = 5

individuals_per_generation = 20
individuals_to_percevere = 4
mutation_rate = 0.3
crossovers_per_generation = 8

snapshot_step = None
snapshot_epoch = False

# Initial generation
current_generation_individuals = get_empty_generation(individuals_per_generation, input_size, output_size, first_fully_connected_size, second_fully_connected_size, activation)


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
		time.sleep(0.2)

	best_performance = max(generation_scores)
	best_individual = generation_scores.index(best_performance)

	mean_score = mean(generation_scores)

	print '--------------------------'
	print 'Best score in this generation: ' + str(best_performance)
	print 'Best individual: ' + str(best_individual)
	print 'Mean performance in the generation: ' + str(mean_score)
	print 'Median performance in the generation: ' + str(median(generation_scores))

	print '==========================================='
	print 'Done with this generation'

	# top_individuals_values = heapq.nlargest(individuals_to_percevere, generation_scores)
	top_individuals_values = heapq.nlargest(individuals_to_percevere-1, generation_scores)

	top_individuals_values.append(generation_scores[randint(0, individuals_per_generation-1)])
	top_individuals_indexes = [generation_scores.index(value) for value in top_individuals_values]

	print 'Perseverent scores: ' + str(top_individuals_values)

	crossovers_so_far = 0
	for index, individual in enumerate(current_generation_individuals):
		if index not in top_individuals_indexes:
			if crossovers_so_far >= crossovers_per_generation:
				print 'Now mutating...'
				heritage = top_individuals_indexes[randint(0, individuals_to_percevere-1)]
				current_generation_individuals[index].clone_individual(current_generation_individuals[heritage])
				current_generation_individuals[index].generate_mutations(mutation_rate)
				# current_generation_individuals[index].generate_mutations(mutation_rate/(mean_score/1.5))
			else:
				print 'Now crossing over...'
				heritage = top_individuals_indexes[randint(0, individuals_to_percevere-1)]
				current_generation_individuals[index].clone_individual(current_generation_individuals[heritage])
				second_individual_to_breed_index = randint(0, individuals_to_percevere-1)
				while second_individual_to_breed_index == heritage:
					second_individual_to_breed_index = randint(0, individuals_to_percevere-1)
				current_generation_individuals[index].generate_crossover(current_generation_individuals[second_individual_to_breed_index])
				crossovers_so_far = crossovers_so_far + 1

	shuffle(current_generation_individuals)

	if os.path.exists(generation_report):
		append_write = 'a'
	else:
		append_write = 'w'
	highscore = open(generation_report, append_write)
	highscore.write("Generation_" + str(current_generation) + ';' + str(top_individuals_values) + ';' + str(mean_score) + '\n')
	highscore.close()

	print '==========================================='
	current_generation = current_generation + 1





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