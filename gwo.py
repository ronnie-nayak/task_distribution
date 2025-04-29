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

def gwo(folder_path, w1=0.5, w2=0.5, population_size=30, iterations=100):
    exec_df = pd.read_csv(os.path.join(folder_path, "ExecutionTable.csv"), index_col=0)
    cost_df = pd.read_csv(os.path.join(folder_path, "CostTable.csv"), index_col=0)

    num_tasks = len(exec_df.columns)
    num_nodes = len(exec_df.index)

    population = [np.random.randint(0, num_nodes, num_tasks) for _ in range(population_size)]
    alpha, beta, delta = None, None, None
    alpha_score, beta_score, delta_score = float("inf"), float("inf"), float("inf")
    alpha_fit, beta_fit, delta_fit = None, None, None

    time_history = []
    cost_history = []

    for l in range(iterations):
        for sol in population:
            fitness, t, c = evaluate(sol, exec_df, cost_df, w1, w2)
            if fitness < alpha_score:
                delta_score, delta, delta_fit = beta_score, beta, beta_fit
                beta_score, beta, beta_fit = alpha_score, alpha, alpha_fit
                alpha_score, alpha, alpha_fit = fitness, sol.copy(), (t, c)
            elif fitness < beta_score:
                delta_score, delta, delta_fit = beta_score, beta, beta_fit
                beta_score, beta, beta_fit = fitness, sol.copy(), (t, c)
            elif fitness < delta_score:
                delta_score, delta, delta_fit = fitness, sol.copy(), (t, c)

        a = 2 - l * (2 / iterations)
        new_population = []
        for sol in population:
            new_sol = np.zeros(num_tasks, dtype=int)
            for i in range(num_tasks):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[i] - sol[i])
                X1 = alpha[i] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[i] - sol[i])
                X2 = beta[i] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[i] - sol[i])
                X3 = delta[i] - A3 * D_delta

                new_val = int((X1 + X2 + X3) / 3.0)
                new_sol[i] = max(0, min(num_nodes - 1, new_val))
            new_population.append(new_sol)

        population = new_population

        time_history.append(alpha_fit[0])
        cost_history.append(alpha_fit[1])

    final_solution = [int(x) for x in alpha]
    return final_solution, alpha_score, alpha_fit[0], alpha_fit[1], time_history, cost_history

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder = "task40"
    solution, fitness, time_val, cost_val, time_hist, cost_hist = gwo(folder)
    print("Best Node Assignment:", solution)
    print(f"Total Execution Time (s): {time_val:.2f}")
    print(f"Total Cost: {cost_val:.2f}")
    print(f"Weighted Fitness: {fitness:.2f}")

    
    plt.figure(figsize=(10, 5))
    plt.plot(time_hist, label='Execution Time (s)', color='blue')
    plt.plot(cost_hist, label='Cost', color='green')
    plt.title('GWO Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
