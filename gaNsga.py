import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def load_data(folder):
    task_details = pd.read_csv(f"{folder}/TaskDetails.csv")
    node_details = pd.read_csv(f"{folder}/NodeDetails.csv")
    execution_table = pd.read_csv(f"{folder}/ExecutionTable.csv")
    cost_table = pd.read_csv(f"{folder}/CostTable.csv")

    task_details.columns = task_details.columns.str.strip()
    node_details.columns = node_details.columns.str.strip()
    execution_table.columns = execution_table.columns.str.strip()
    cost_table.columns = cost_table.columns.str.strip()

    task_list = task_details['TaskID'].tolist()
    node_list = node_details['NodeID'].tolist()
    num_tasks = len(task_list)
    num_nodes = len(node_list)

    exec_dict = execution_table.set_index('NodeID').T.to_dict()
    cost_dict = cost_table.set_index('NodeID').T.to_dict()

    return task_list, node_list, num_tasks, num_nodes, exec_dict, cost_dict


POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 2


def evaluate(individual, task_list, node_list, exec_dict, cost_dict):
    time = 0
    cost = 0
    for i, node_idx in enumerate(individual):
        task = task_list[i]
        node = node_list[node_idx]
        time += exec_dict[node][task]
        cost += cost_dict[node][task]
    return time, cost

def dominates(a, b):
    return (a[0] <= b[0] and a[1] < b[1]) or (a[0] < b[0] and a[1] <= b[1])

def initial_population(num_tasks, num_nodes):
    return [np.random.randint(0, num_nodes, num_tasks).tolist() for _ in range(POP_SIZE)]

def crossover(p1, p2, num_tasks):
    point = random.randint(1, num_tasks - 2)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(ind, num_nodes, rate=MUTATION_RATE):
    for i in range(len(ind)):
        if random.random() < rate:
            ind[i] = random.randint(0, num_nodes - 1)
    return ind


def fast_nondominated_sort(pop, fitnesses):
    S, n, rank = {}, {}, {}
    fronts = [[]]

    for p in range(len(pop)):
        S[p], n[p] = [], 0
        for q in range(len(pop)):
            if dominates(fitnesses[p], fitnesses[q]):
                S[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]

def crowding_distance(front, fitnesses):
    distance = [0] * len(front)
    for m in range(2):
        obj_values = [fitnesses[i][m] for i in front]
        sorted_indices = np.argsort(obj_values)
        max_val, min_val = max(obj_values), min(obj_values)
        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = float('inf')
        for i in range(1, len(front) - 1):
            prev = obj_values[sorted_indices[i - 1]]
            next_ = obj_values[sorted_indices[i + 1]]
            distance[sorted_indices[i]] += (next_ - prev) / (max_val - min_val + 1e-9)
    return distance

def select(pop, fitnesses):
    fronts = fast_nondominated_sort(pop, fitnesses)
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) > POP_SIZE:
            distances = crowding_distance(front, fitnesses)
            sorted_front = [x for _, x in sorted(zip(distances, front), reverse=True)]
            new_pop.extend([pop[i] for i in sorted_front[:POP_SIZE - len(new_pop)]])
            break
        new_pop.extend([pop[i] for i in front])
    return new_pop

def nsga(folder):
    task_list, node_list, num_tasks, num_nodes, exec_dict, cost_dict = load_data(folder)
    population = initial_population(num_tasks, num_nodes)
    fitnesses = [evaluate(ind, task_list, node_list, exec_dict, cost_dict) for ind in population]

    cloud_fog_over_time = []  

    for gen in range(GENERATIONS):
        
        gen_cloud = 0
        gen_fog = 0
        for ind in population:
            for node_idx in ind:
                if node_idx < 3:
                    gen_cloud += 1
                else:
                    gen_fog += 1
        cloud_fog_over_time.append((gen_cloud, gen_fog))

        
        offspring = []
        while len(offspring) < POP_SIZE:
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover(p1, p2, num_tasks)
            offspring.append(mutate(c1, num_nodes))
            if len(offspring) < POP_SIZE:
                offspring.append(mutate(c2, num_nodes))
        offspring_fitness = [evaluate(ind, task_list, node_list, exec_dict, cost_dict) for ind in offspring]
        combined = population + offspring
        combined_fitness = fitnesses + offspring_fitness
        population = select(combined, combined_fitness)
        fitnesses = [evaluate(ind, task_list, node_list, exec_dict, cost_dict) for ind in population]

    return fitnesses, cloud_fog_over_time

if __name__ == "__main__":
    folder = "task40"
    fitnesses = nsga(folder)
    times = [f[0] for f in fitnesses]
    costs = [f[1] for f in fitnesses]

    plt.figure(figsize=(10, 6))
    plt.scatter(times, costs, color='blue', s=40)
    plt.xlabel("Execution Time")
    plt.ylabel("Execution Cost")
    plt.title("NSGA-II Pareto Front")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
