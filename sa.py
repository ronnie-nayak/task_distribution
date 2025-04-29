import pandas as pd
import numpy as np
import os
import random
import math

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

def sa(folder_path, w1=0.5, w2=0.5, max_iter=1000, initial_temp=1000, cooling_rate=0.95):
    exec_df = pd.read_csv(os.path.join(folder_path, "ExecutionTable.csv"), index_col=0)
    cost_df = pd.read_csv(os.path.join(folder_path, "CostTable.csv"), index_col=0)

    num_tasks = len(exec_df.columns)
    num_nodes = len(exec_df.index)

    current_solution = [random.randint(0, num_nodes - 1) for _ in range(num_tasks)]
    current_score, current_time, current_cost = evaluate(current_solution, exec_df, cost_df, w1, w2)

    best_solution = current_solution[:]
    best_score = current_score
    best_time = current_time
    best_cost = current_cost

    time_history = [best_time]
    cost_history = [best_cost]

    temp = initial_temp

    for _ in range(max_iter):
        neighbor = current_solution[:]
        task_to_change = random.randint(0, num_tasks - 1)
        neighbor[task_to_change] = random.randint(0, num_nodes - 1)

        new_score, new_time, new_cost = evaluate(neighbor, exec_df, cost_df, w1, w2)

        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_solution = neighbor[:]
            current_score = new_score
            current_time = new_time
            current_cost = new_cost

        if current_score < best_score:
            best_solution = current_solution[:]
            best_score = current_score
            best_time = current_time
            best_cost = current_cost

        time_history.append(best_time)
        cost_history.append(best_cost)

        temp *= cooling_rate

    final_solution = [int(x) for x in best_solution]
    return final_solution, best_score, best_time, best_cost, time_history, cost_history

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder = "task40"
    solution, fitness, time_val, cost_val, time_hist, cost_hist = sa(folder)
    print("Best Node Assignment:", solution)
    print(f"Total Execution Time (s): {time_val:.2f}")
    print(f"Total Cost: {cost_val:.2f}")
    print(f"Weighted Fitness: {fitness:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(time_hist, label='Execution Time (s)', color='blue')
    plt.plot(cost_hist, label='Cost', color='green')
    plt.title('Simulated Annealing Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
