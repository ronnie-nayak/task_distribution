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
    
    exec_dict = execution_table.set_index('NodeID').T.to_dict()
    cost_dict = cost_table.set_index('NodeID').T.to_dict()

    return task_list, node_list, exec_dict, cost_dict


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


def mogwo(folder, pop_size=30, generations=100):
    task_list, node_list, exec_dict, cost_dict = load_data(folder)
    num_tasks = len(task_list)
    num_nodes = len(node_list)

    population = [np.random.randint(0, num_nodes, num_tasks).tolist() for _ in range(pop_size)]
    fitnesses = [evaluate(ind, task_list, node_list, exec_dict, cost_dict) for ind in population]

    alpha, beta, delta = population[0], population[1], population[2]
    alpha_score, beta_score, delta_score = fitnesses[0], fitnesses[1], fitnesses[2]

    cloud_fog_over_time = []

    for gen in range(generations):
        
        gen_cloud = 0
        gen_fog = 0
        for ind in population:
            for node_idx in ind:
                if node_idx < 3:
                    gen_cloud += 1
                else:
                    gen_fog += 1
        cloud_fog_over_time.append((gen_cloud, gen_fog))

        a = 2 - gen * (2 / generations)
        new_population = []

        for i in range(pop_size):
            wolf = []
            for j in range(num_tasks):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - population[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - population[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - population[i][j])
                X3 = delta[j] - A3 * D_delta

                pos = int((X1 + X2 + X3) / 3)
                wolf.append(np.clip(pos, 0, num_nodes - 1))

            new_population.append(wolf)

        combined_population = population + new_population
        combined_fitnesses = [evaluate(ind, task_list, node_list, exec_dict, cost_dict) for ind in combined_population]

        fronts = fast_nondominated_sort(combined_population, combined_fitnesses)
        population = []
        for front in fronts:
            if len(population) + len(front) <= pop_size:
                population += [combined_population[i] for i in front]
            else:
                remaining = pop_size - len(population)
                population += [combined_population[i] for i in front[:remaining]]
                break

        fitnesses = [evaluate(ind, task_list, node_list, exec_dict, cost_dict) for ind in population]
        sorted_fit = sorted(zip(fitnesses, population), key=lambda x: x[0])
        alpha, beta, delta = sorted_fit[0][1], sorted_fit[1][1], sorted_fit[2][1]
        alpha_score, beta_score, delta_score = sorted_fit[0][0], sorted_fit[1][0], sorted_fit[2][0]

        print(f"Generation {gen + 1}: Alpha Time = {alpha_score[0]:.2f}, Alpha Cost = {alpha_score[1]:.2f}")

    return fitnesses, cloud_fog_over_time


if __name__ == "__main__":
    fitnesses = mogwo("task40")
    times = [f[0] for f in fitnesses]
    costs = [f[1] for f in fitnesses]

    plt.figure(figsize=(10, 6))
    plt.scatter(times, costs, color='indigo', s=40)
    plt.xlabel("Execution Time")
    plt.ylabel("Execution Cost")
    plt.title("MOGWO Pareto Front")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
