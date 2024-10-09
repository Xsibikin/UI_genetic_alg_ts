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


initial_temp = 100
cooling_rate = 0.9999
stopping_temp = 1

def distance(city1, city2):
	return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def fitness(route):
	td = 0
	for i in range(len(route) - 1):
		td += distance(Cities[route[i]], Cities[route[i + 1]])
	total_distance = td
	total_distance += distance(Cities[route[-1]], Cities[route[0]])
	return total_distance


def initial_solution():
	individual = list(range(len(Cities)))
	random.shuffle(individual)
	return individual



def generate_neighbor(solution):

    roulette = random.uniform(0, 1)

    if roulette < 0.5:

        neighbor = solution.copy()
        start, end = sorted(random.sample(range(len(solution)), 2))

        insertion_place = random.randint(0, len(solution) - (end - start) - 1)
        route_piece = neighbor[start:end + 1]
        del neighbor[start:end + 1]

        neighbor[insertion_place:insertion_place] = route_piece
    else:
        neighbor = solution.copy()
        i = random.randint(0, len(solution) - 2)
        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]

    return neighbor


def simulated_annealing(initial_temp, cooling_rate, stopping_temp):
    current_solution = initial_solution()
    current_distance = fitness(current_solution)

    best_solution = current_solution
    best_distance = current_distance

    temperature = initial_temp

    distances_per_iteration = [current_distance]
    iterations = 0

    while temperature > stopping_temp:
        neighbor = generate_neighbor(current_solution)
        neighbor_distance = fitness(neighbor)


        if neighbor_distance < current_distance:
            current_solution = neighbor
            current_distance = neighbor_distance


            if neighbor_distance < best_distance:
                best_solution = neighbor
                best_distance = neighbor_distance

        else:
            probability = np.exp((current_distance - neighbor_distance) / temperature)
            if random.random() < probability:
                current_solution = neighbor
                current_distance = neighbor_distance

        iterations += 1
        temperature *= cooling_rate

        distances_per_iteration.append(current_distance)

    return best_solution, best_distance, distances_per_iteration, iterations


def plot_route(best_solution, best_distance, distances_per_iteration, iterations):

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    x = [Cities[i][0] for i in best_solution] + [Cities[best_solution[0]][0]]
    y = [Cities[i][1] for i in best_solution] + [Cities[best_solution[0]][1]]

    plt.plot(x, y, 'o-', label='Route')
    plt.scatter([c[0] for c in Cities], [c[1] for c in Cities], c='red')

    for i, (xi, yi) in enumerate(Cities):
        plt.text(xi, yi, f"City {i + 1}", fontsize=10)

    plt.title(f'Best Route Found by Simulated Annealing: {best_distance}\nIterations: {iterations}', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(distances_per_iteration, color='blue', label='Distance per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Simulated Annealing Progress', fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


best_solution, best_distance, distances_per_iteration, iterations = simulated_annealing(initial_temp, cooling_rate, stopping_temp)
plot_route(best_solution, best_distance, distances_per_iteration, iterations)