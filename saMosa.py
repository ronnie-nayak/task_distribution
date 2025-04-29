
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




def multi_objective_sa(folder, initial_temp=100, cooling_rate=0.95, iterations=1000):
    task_list, node_list, exec_dict, cost_dict = load_data(folder)
    num_tasks = len(task_list)
    num_nodes = len(node_list)

    def evaluate(solution):
        time = sum(exec_dict[node_list[node_idx]][task_list[i]] for i, node_idx in enumerate(solution))
        cost = sum(cost_dict[node_list[node_idx]][task_list[i]] for i, node_idx in enumerate(solution))
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

    current_solution = [random.randint(0, num_nodes - 1) for _ in range(num_tasks)]
    current_fitness = evaluate(current_solution)
    archive = [{'solution': current_solution, 'fitness': current_fitness}]
    temp = initial_temp
    cloud_fog_over_time = []

    for it in range(iterations):
        neighbor = current_solution[:]
        idx = random.randint(0, num_tasks - 1)
        neighbor[idx] = random.randint(0, num_nodes - 1)
        neighbor_fitness = evaluate(neighbor)

        delta = sum(neighbor_fitness) - sum(current_fitness)
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_solution = neighbor
            current_fitness = neighbor_fitness
            archive = update_archive(archive, neighbor, neighbor_fitness)

        
        cloud = sum(1 for node in current_solution if node < 3)
        fog = num_tasks - cloud
        cloud_fog_over_time.append((cloud, fog))

        temp *= cooling_rate

        if (it + 1) % 100 == 0:
            print(f"Iteration {it + 1}: Archive size = {len(archive)}")

    return archive, cloud_fog_over_time




if __name__ == "__main__":
    folder = "task40"
    pareto_archive, _ = multi_objective_sa(folder)

    pareto_front = np.array([p['fitness'] for p in pareto_archive])
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='orange')
    plt.xlabel("Total Execution Time")
    plt.ylabel("Total Cost")
    plt.title("Pareto Front - Multi-Objective SA")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
