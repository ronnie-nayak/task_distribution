import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

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

def gwo(folder_path, allowed_nodes, w1=0.5, w2=0.5, population_size=30, iterations=100):
    exec_df = pd.read_csv(os.path.join(folder_path, "ExecutionTable.csv"), index_col=0)
    cost_df = pd.read_csv(os.path.join(folder_path, "CostTable.csv"), index_col=0)

    exec_df = exec_df.iloc[allowed_nodes]
    cost_df = cost_df.iloc[allowed_nodes]

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
    folder = "task40"
    scenarios = {
        "Cloud Only": list(range(0, 3)),
        "Fog Only": list(range(3, 13)),
        "Hybrid": list(range(0, 13))
    }

    results = {}

    for label, nodes in scenarios.items():
        sol, fit, t, c, th, ch = gwo(folder, nodes)
        results[label] = {
            "solution": sol,
            "fitness": fit,
            "time": t,
            "cost": c,
            "time_hist": th,
            "cost_hist": ch
        }

    for label, res in results.items():
        print(f"\n=== {label} ===")
        print("Best Node Assignment:", res["solution"])
        print(f"Total Execution Time: {res['time']:.2f} s")
        print(f"Total Cost: {res['cost']:.2f}")
        print(f"Weighted Fitness: {res['fitness']:.2f}")

    
    plt.figure(figsize=(10, 5))
    colors = {"Cloud Only": "green", "Fog Only": "navy", "Hybrid": "darkorange"}
    for label in ["Fog Only", "Hybrid", "Cloud Only"]:
        res = results[label]
        plt.fill_between(range(len(res["time_hist"])), res["time_hist"], alpha=1, label=label, color=colors[label])
    plt.title("Execution Time (Area Under Curve)")
    plt.xlabel("Generation")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    plt.figure(figsize=(10, 5))
    for label in ["Cloud Only", "Hybrid", "Fog Only"]:
        res = results[label]
        plt.fill_between(range(len(res["cost_hist"])), res["cost_hist"], alpha=1, label=label, color=colors[label])
    plt.title("Cost (Area Under Curve)")
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
