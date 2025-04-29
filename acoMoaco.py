import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os




def load_data(folder):
    task_details = pd.read_csv(os.path.join(folder, "TaskDetails.csv"))
    node_details = pd.read_csv(os.path.join(folder, "NodeDetails.csv"))
    execution_table = pd.read_csv(os.path.join(folder, "ExecutionTable.csv"))
    cost_table = pd.read_csv(os.path.join(folder, "CostTable.csv"))

    task_details.columns = task_details.columns.str.strip()
    node_details.columns = node_details.columns.str.strip()
    execution_table.columns = execution_table.columns.str.strip()
    cost_table.columns = cost_table.columns.str.strip()

    task_list = task_details['TaskID'].tolist()
    node_list = node_details['NodeID'].tolist()
    exec_dict = execution_table.set_index('NodeID').T.to_dict()
    cost_dict = cost_table.set_index('NodeID').T.to_dict()

    return task_list, node_list, exec_dict, cost_dict




def multi_objective_aco(folder, num_ants=30, num_iterations=100, alpha=1.0, beta=2.0, rho=0.1):
    task_list, node_list, exec_dict, cost_dict = load_data(folder)
    num_tasks = len(task_list)
    num_nodes = len(node_list)
    pheromone = np.ones((num_tasks, num_nodes))
    pareto_archive = []
    cloud_fog_over_time = []

    def evaluate(solution):
        time, cost = 0, 0
        for task_idx, node_idx in enumerate(solution):
            task = task_list[task_idx]
            node = node_list[node_idx]
            time += exec_dict[node][task]
            cost += cost_dict[node][task]
        return time, cost

    def dominates(a, b):
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def update_archive(archive, solution, fitness):
        non_dominated = []
        for a in archive:
            if dominates(fitness, a['fitness']):
                continue
            elif dominates(a['fitness'], fitness):
                return archive
            else:
                non_dominated.append(a)
        non_dominated.append({'solution': solution, 'fitness': fitness})
        return non_dominated

    for iteration in range(num_iterations):
        all_solutions = []
        gen_cloud, gen_fog = 0, 0

        for ant in range(num_ants):
            solution = []
            for task_idx in range(num_tasks):
                probabilities = []
                for node_idx in range(num_nodes):
                    time = exec_dict[node_list[node_idx]][task_list[task_idx]]
                    cost = cost_dict[node_list[node_idx]][task_list[task_idx]]
                    heuristic = 1.0 / (time + cost + 1e-6)
                    prob = (pheromone[task_idx][node_idx] ** alpha) * (heuristic ** beta)
                    probabilities.append(prob)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                chosen_node = np.random.choice(range(num_nodes), p=probabilities)
                solution.append(chosen_node)
                if chosen_node < 3:
                    gen_cloud += 1
                else:
                    gen_fog += 1

            fitness = evaluate(solution)
            pareto_archive = update_archive(pareto_archive, solution, fitness)
            all_solutions.append((solution, fitness))

        cloud_fog_over_time.append((gen_cloud, gen_fog))

        pheromone *= (1 - rho)
        for sol, fit in all_solutions:
            for task_idx, node_idx in enumerate(sol):
                delta = 1 / (fit[0] + fit[1] + 1e-6)
                pheromone[task_idx][node_idx] += delta

        print(f"Iteration {iteration + 1}: Pareto front size = {len(pareto_archive)}")

    return pareto_archive, cloud_fog_over_time

if __name__ == "__main__":
    pareto_archive, _ = multi_objective_aco("task40")

    
    pareto_front = np.array([p['fitness'] for p in pareto_archive])
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='green')
    plt.xlabel("Total Execution Time")
    plt.ylabel("Total Cost")
    plt.title("Pareto Front - Multi-Objective ACO")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
