import numpy as np
import matplotlib.pyplot as plt
import random

# Заранее определенные координаты 40 городов
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

# Параметры генетического алгоритма
population_size = 800  # Размер популяции
generations = 400  # Количество поколений
mutation_rate = 0.02  # Вероятность мутации (1-2%)



# Функция для вычисления расстояния между двумя точками
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# Фитнесс-функция (суммарное расстояние по маршруту)
def fitness(route):
    total_distance = sum(distance(Cities[route[i]], Cities[route[i + 1]]) for i in range(len(route) - 1))
    total_distance += distance(Cities[route[-1]], Cities[route[0]])  # Замыкаем маршрут
    return total_distance


# Генерация начальной популяции
def initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = list(range(len(Cities)))
        random.shuffle(individual)
        population.append(individual)
    return population


# Отбор по турнирной селекции
def tournament_selection(population, fitness_values):
    tournament_size = 5
    selected = random.sample(list(zip(population, fitness_values)), tournament_size)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]  # Возвращаем особь с наилучшей фитнесс-функцией


# Отбор по рулетке (пропорционально фитнессу)
def roulette_selection(population, fitness_values):
    total_fitness = sum(1 / f for f in fitness_values)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness_val in zip(population, fitness_values):
        current += (1 / fitness_val)
        if current > pick:
            return individual


# Кроссовер двух родителей
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end + 1] = parent1[start:end + 1]

    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city
    return child


# Мутация (обмен местами двух соседних городов)
def mutate(individual):
    if random.random() < mutation_rate:
        i = random.randint(0, len(individual) - 2)
        individual[i], individual[i + 1] = individual[i + 1], individual[i]


# Основная функция генетического алгоритма
def genetic_algorithm():
    population = initial_population(population_size)
    helper = 9999999
    for generation in range(generations):
        fitness_values = [fitness(ind) for ind in population]

        new_population = []
        for _ in range(population_size):
            # Смешиваем два метода выбора родителей

            if random.random() < 0.5:
                parent1 = tournament_selection(population, fitness_values)
            else:
                parent1 = roulette_selection(population, fitness_values)

            if random.random() < 0.5:
                parent2 = tournament_selection(population, fitness_values)
            else:
                parent2 = roulette_selection(population, fitness_values)


            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

        population = new_population

        # Выводим прогресс каждые 10 поколений
        if fitness_values[generation] < helper:
            helper = fitness_values[generation]
            print(f"Generation {generation}, Best distance: {fitness_values[generation]}")

    # Возвращаем лучшее решение
    best_route = population[fitness_values.index(min(fitness_values))]
    return best_route, min(fitness_values)


# Функция для визуализации маршрута
# def plot_route(route):
#     plt.figure(figsize=(8, 8))
#     x = [Cities[i][0] for i in route] + [Cities[route[0]][0]]
#     y = [Cities[i][1] for i in route] + [Cities[route[0]][1]]
#     plt.plot(x, y, 'o-', label='Route')
#     plt.scatter([c[0] for c in Cities], [c[1] for c in Cities], c='red')
#     for i, (xi, yi) in enumerate(Cities):
#         plt.text(xi, yi, f"City {i + 1}", fontsize=12)
#     plt.title('Best Route Found by Genetic Algorithm')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def plot_route(route, best_distance):
    plt.figure(figsize=(8, 8))
    x = [Cities[i][0] for i in route] + [Cities[route[0]][0]]
    y = [Cities[i][1] for i in route] + [Cities[route[0]][1]]

    plt.plot(x, y, 'o-', label='Route')
    plt.scatter([c[0] for c in Cities], [c[1] for c in Cities], c='red')

    for i, (xi, yi) in enumerate(Cities):
        plt.text(xi, yi, f"City {i + 1}", fontsize=12)

    # Добавление текста для итогового пути
    plt.title(f'Best Route Found by Genetic Algorithm: {best_distance}', fontsize=10)

    plt.legend()
    plt.grid(True)
    plt.show()


# Запуск алгоритма и визуализация лучшего маршрута
best_route, best_distance = genetic_algorithm()
plot_route(best_route, best_distance)
