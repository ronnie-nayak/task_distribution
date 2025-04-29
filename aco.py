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

def aco(folder_path, w1=0.5, w2=0.5, num_ants=30, iterations=100, alpha=1, beta=2, rho=0.1, Q=1.0):
    exec_df = pd.read_csv(os.path.join(folder_path, "ExecutionTable.csv"), index_col=0)
    cost_df = pd.read_csv(os.path.join(folder_path, "CostTable.csv"), index_col=0)

    num_tasks = len(exec_df.columns)
    num_nodes = len(exec_df.index)

    pheromone = np.ones((num_tasks, num_nodes))
    visibility = np.zeros((num_tasks, num_nodes))

    for task_idx in range(num_tasks):
        for node_idx in range(num_nodes):
            exec_time = exec_df.iloc[node_idx, task_idx]
            cost = cost_df.iloc[node_idx, task_idx]
            visibility[task_idx][node_idx] = 1.0 / (w1 * exec_time + w2 * cost)

    best_solution = None
    best_score = float('inf')
    best_time = 0.0
    best_cost = 0.0
    time_history = []
    cost_history = []

    for it in range(iterations):
        all_solutions = []
        all_scores = []

        for ant in range(num_ants):
            solution = []
            for task_idx in range(num_tasks):
                probs = (pheromone[task_idx] ** alpha) * (visibility[task_idx] ** beta)
                probs /= probs.sum()
                node_idx = np.random.choice(range(num_nodes), p=probs)
                solution.append(node_idx)

            score, total_time, total_cost = evaluate(solution, exec_df, cost_df, w1, w2)
            all_solutions.append((solution, score, total_time, total_cost))
            all_scores.append(score)

        pheromone *= (1 - rho)

        for solution, score, _, _ in all_solutions:
            for task_idx, node_idx in enumerate(solution):
                pheromone[task_idx][node_idx] += Q / score

        iteration_best = min(all_solutions, key=lambda x: x[1])
        if iteration_best[1] < best_score:
            best_solution = iteration_best[0]
            best_score = iteration_best[1]
            best_time = iteration_best[2]
            best_cost = iteration_best[3]

        time_history.append(best_time)
        cost_history.append(best_cost)

    final_solution = [int(x) for x in best_solution]
    return final_solution, best_score, best_time, best_cost, time_history, cost_history

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder = "task40"
    solution, fitness, time_val, cost_val, time_hist, cost_hist = aco(folder)
    print("Best Node Assignment:", solution)
    print(f"Total Execution Time (s): {time_val:.2f}")
    print(f"Total Cost: {cost_val:.2f}")
    print(f"Weighted Fitness: {fitness:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(time_hist, label='Execution Time (s)', color='blue')
    plt.plot(cost_hist, label='Cost', color='green')
    plt.title('ACO Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
