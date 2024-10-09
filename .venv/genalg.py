import numpy as np
import matplotlib.pyplot as plt
import random

Cities = [
	(69, 55), (63, 148), (135, 144), (74, 48), (30, 26),
	(148, 67), (151, 183), (80, 196), (194, 92), (141, 152),
	(73, 78), (82, 17), (60, 159), (142, 159), (96, 24),
	(97, 6), (168, 80), (78, 179), (175, 49), (42, 138),
	(83, 116), (153, 89), (137, 60), (45, 5), (91, 150),
	(168, 190), (194, 116), (50, 196), (150, 10), (10, 100),
	(150, 50), (190, 150), (80, 160), (120, 50), (110, 10),
	(10, 190), (190, 10), (10, 10), (190, 190), (50, 50)
]

population_size = 800
generations = 400
mutation_rate = 0.02
tournament_size = 5



def distance(city1, city2):
	return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def fitness(route):
	td = 0
	for i in range(len(route) - 1):
		td += distance(Cities[route[i]], Cities[route[i + 1]])
	total_distance = td
	total_distance += distance(Cities[route[-1]], Cities[route[0]])
	return total_distance


def generate_population(population_size):
	population = []
	for _ in range(population_size):
		individual = list(range(len(Cities)))
		random.shuffle(individual)
		population.append(individual)
	return population


def tournament_selection(population, fitness_values):

	fitness_dict = list(zip(population, fitness_values))
	tournament_dict = random.sample(fitness_dict, tournament_size)
	tournament_dict.sort(key=lambda x: x[1])

	return tournament_dict[0][0]


def roulette_selection(population, fitness_values):

	total_fitness = 0
	for f in fitness_values:
		total_fitness += 1 / f
	roulette_value = random.uniform(0, total_fitness)

	current_value = 0
	for individual, fitness_value in zip(population, fitness_values):
		current_value += (1 / fitness_value)
		if current_value > roulette_value:
			return individual


def crossover(parent_1, parent_2):

	start, end = sorted(random.sample(range(len(parent_1)), 2))
	child = [-1] * len(parent_1)
	child[start:end+1] = parent_1[start:end+1]

	pointer = 0
	for city in parent_2:
		if city not in child:
			while child[pointer] != -1:
				pointer += 1
			child[pointer] = city
	return child


def mutate(individual):
	if random.random() < mutation_rate:
		i = random.randint(0, len(individual) - 2)
		individual[i], individual[i + 1] = individual[i + 1], individual[i]



def genetic_algorithm():
	population = generate_population(population_size)
	helper = 999999999

	for generation in range(generations):

		fitness_values = []
		for index in population:
			fitness_values.append(fitness(index))

		new_population = []
		for _ in range(population_size):

			if random.random() < 0.5:
				parent_1 = tournament_selection(population, fitness_values)
			else:
				parent_1 = roulette_selection(population, fitness_values)

			if random.random() < 0.5:
				parent_2 = tournament_selection(population, fitness_values)
			else:
				parent_2 = roulette_selection(population, fitness_values)

			child = crossover(parent_1, parent_2)
			mutate(child)
			new_population.append(child)

		population = new_population

		if fitness_values[generation] < helper:
			helper = fitness_values[generation]
			print(f"Generation {generation}, Best distance: {fitness_values[generation]}")


	best_route = population[fitness_values.index(min(fitness_values))]
	return best_route, min(fitness_values)


def plot_route(route, best_distance):
	plt.figure(figsize=(8, 8))
	x = [Cities[i][0] for i in route] + [Cities[route[0]][0]]
	y = [Cities[i][1] for i in route] + [Cities[route[0]][1]]

	plt.plot(x, y, 'o-', label='Route')
	plt.scatter([c[0] for c in Cities], [c[1] for c in Cities], c='red')

	for i, (xi, yi) in enumerate(Cities):
		plt.text(xi, yi, f"City {i + 1}", fontsize=12)

	plt.title(f'Best Route Found by Genetic Algorithm: {best_distance}', fontsize=10)

	plt.legend()
	plt.grid(True)
	plt.show()



best_route, best_distance = genetic_algorithm()
plot_route(best_route, best_distance)