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




def multi_objective_pso(folder, num_particles=30, num_iterations=100, w=0.5, c1=1.5, c2=1.5):
    task_list, node_list, exec_dict, cost_dict = load_data(folder)
    num_tasks = len(task_list)
    num_nodes = len(node_list)

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
        non_dominated.append({'solution': solution.copy(), 'fitness': fitness})
        return non_dominated

    particles = [np.random.randint(0, num_nodes, num_tasks).tolist() for _ in range(num_particles)]
    velocities = [np.zeros(num_tasks) for _ in range(num_particles)]
    pbest = particles.copy()
    pbest_fitness = [evaluate(p) for p in particles]
    archive = []
    cloud_fog_over_time = []

    for iteration in range(num_iterations):
        gen_cloud, gen_fog = 0, 0

        for i in range(num_particles):
            fitness = evaluate(particles[i])
            if dominates(fitness, pbest_fitness[i]):
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness
            archive = update_archive(archive, particles[i], fitness)

            
            for node_idx in particles[i]:
                if node_idx < 3:
                    gen_cloud += 1
                else:
                    gen_fog += 1

        cloud_fog_over_time.append((gen_cloud, gen_fog))

        for i in range(num_particles):
            leader = random.choice(archive)['solution'] if archive else pbest[i]
            for d in range(num_tasks):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (w * velocities[i][d] +
                                    c1 * r1 * (pbest[i][d] - particles[i][d]) +
                                    c2 * r2 * (leader[d] - particles[i][d]))
                particles[i][d] = int(np.clip(round(particles[i][d] + velocities[i][d]), 0, num_nodes - 1))

        print(f"Iteration {iteration + 1}: Pareto front size = {len(archive)}")

    return archive, cloud_fog_over_time

if __name__ == "__main__":
    pareto_archive, _ = multi_objective_pso("task40")

    
    pareto_front = np.array([p['fitness'] for p in pareto_archive])
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red')
    plt.xlabel("Total Execution Time")
    plt.ylabel("Total Cost")
    plt.title("Pareto Front - Multi-Objective PSO")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
