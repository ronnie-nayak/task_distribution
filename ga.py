import pandas as pd
import numpy as np
import os
import random

def evaluate(solution, exec_df, cost_df, w1, w2):
    total_time = 0.0
    total_cost = 0.0
    for task_idx, node_idx in enumerate(solution):
        node_id = exec_df.index[int(node_idx)]
        task_id = exec_df.columns[task_idx]

        exec_time = exec_df.loc[node_id, task_id]
        cost = cost_df.loc[node_id, task_id]

        total_time += exec_time
        total_cost += cost

    fitness = w1 * total_time + w2 * total_cost
    return fitness, total_time, total_cost

def genetic_algorithm(folder_path, w1=0.5, w2=0.5, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, elite_size=2):
    exec_df = pd.read_csv(os.path.join(folder_path, "ExecutionTable.csv"), index_col=0)
    cost_df = pd.read_csv(os.path.join(folder_path, "CostTable.csv"), index_col=0)

    num_tasks = len(exec_df.columns)
    num_nodes = len(exec_df.index)

    population = [np.random.randint(0, num_nodes, num_tasks) for _ in range(population_size)]
    fitness_history = []
    time_history = []
    cost_history = []

    for generation in range(generations):
        evaluated = [evaluate(individual, exec_df, cost_df, w1, w2) for individual in population]
        fitness_scores = [fit[0] for fit in evaluated]
        time_scores = [fit[1] for fit in evaluated]
        cost_scores = [fit[2] for fit in evaluated]

        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        evaluated = [evaluated[i] for i in sorted_indices]

        fitness_history.append(evaluated[0][0])
        time_history.append(evaluated[0][1])
        cost_history.append(evaluated[0][2])

        new_population = population[:elite_size]

        while len(new_population) < population_size:
            if random.random() < crossover_rate:
                parent1, parent2 = random.sample(population[:20], 2)
                point = random.randint(1, num_tasks - 2)
                child = np.concatenate((parent1[:point], parent2[point:]))
            else:
                child = population[random.randint(0, population_size - 1)].copy()

            for i in range(num_tasks):
                if random.random() < mutation_rate:
                    child[i] = random.randint(0, num_nodes - 1)

            new_population.append(child)

        population = new_population

    best_solution = population[0]
    best_fitness, best_time, best_cost = evaluate(best_solution, exec_df, cost_df, w1, w2)
    final_solution = [int(x) for x in best_solution]

    return final_solution, best_fitness, best_time, best_cost, time_history, cost_history

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder = "task40"
    solution, fitness, time_val, cost_val, time_hist, cost_hist = genetic_algorithm(folder)
    print("Best Node Assignment:", solution)
    print(f"Total Execution Time (s): {time_val:.2f}")
    print(f"Total Cost: {cost_val:.2f}")
    print(f"Weighted Fitness: {fitness:.2f}")

    
    plt.figure(figsize=(10, 5))
    plt.plot(time_hist, label='Execution Time (s)', color='blue')
    plt.plot(cost_hist, label='Cost', color='green')
    plt.title('GA Optimization Progress')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
