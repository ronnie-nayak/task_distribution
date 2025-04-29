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

def pso(folder_path, w1=0.5, w2=0.5, num_particles=30, iterations=100, w=0.5, c1=1.5, c2=1.5):
    exec_df = pd.read_csv(os.path.join(folder_path, "ExecutionTable.csv"), index_col=0)
    cost_df = pd.read_csv(os.path.join(folder_path, "CostTable.csv"), index_col=0)

    num_tasks = len(exec_df.columns)
    num_nodes = len(exec_df.index)

    particles = [np.random.randint(0, num_nodes, num_tasks) for _ in range(num_particles)]
    velocities = [np.zeros(num_tasks) for _ in range(num_particles)]

    pbest = particles[:]
    pbest_scores = [evaluate(p, exec_df, cost_df, w1, w2)[0] for p in particles]

    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score, gbest_time, gbest_cost = evaluate(gbest, exec_df, cost_df, w1, w2)

    time_history = []
    cost_history = []

    for it in range(iterations):
        for i in range(num_particles):
            for t in range(num_tasks):
                r1 = random.random()
                r2 = random.random()
                velocities[i][t] = (w * velocities[i][t] +
                                    c1 * r1 * (pbest[i][t] - particles[i][t]) +
                                    c2 * r2 * (gbest[t] - particles[i][t]))
                particles[i][t] = int(np.clip(round(particles[i][t] + velocities[i][t]), 0, num_nodes - 1))

            score, _, _ = evaluate(particles[i], exec_df, cost_df, w1, w2)
            if score < pbest_scores[i]:
                pbest[i] = particles[i][:]
                pbest_scores[i] = score

        best_local_idx = np.argmin(pbest_scores)
        local_score, local_time, local_cost = evaluate(pbest[best_local_idx], exec_df, cost_df, w1, w2)
        if local_score < gbest_score:
            gbest = pbest[best_local_idx][:]
            gbest_score = local_score
            gbest_time = local_time
            gbest_cost = local_cost

        time_history.append(gbest_time)
        cost_history.append(gbest_cost)

    final_solution = [int(x) for x in gbest]
    return final_solution, gbest_score, gbest_time, gbest_cost, time_history, cost_history

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    folder = "task40"
    solution, fitness, time_val, cost_val, time_hist, cost_hist = pso(folder)
    print("Best Node Assignment:", solution)
    print(f"Total Execution Time (s): {time_val:.2f}")
    print(f"Total Cost: {cost_val:.2f}")
    print(f"Weighted Fitness: {fitness:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(time_hist, label='Execution Time (s)', color='blue')
    plt.plot(cost_hist, label='Cost', color='green')
    plt.title('PSO Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
