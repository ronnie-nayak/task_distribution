import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def load_data(task_details_path, node_details_path, execution_time_path, cost_table_path):
    """
    Load and prepare the data from the CSV files.
    
    Parameters:
    task_details_path: Path to TaskDetails.csv
    node_details_path: Path to NodeDetails.csv
    execution_time_path: Path to ExecutionTable.csv
    cost_table_path: Path to CostTable.csv
    
    Returns:
    tasks_df: DataFrame with task information
    nodes_df: DataFrame with node information
    execution_times_df: DataFrame with execution times for each task on each node
    costs_df: DataFrame with costs for each task on each node
    """
    
    tasks_df = pd.read_csv(task_details_path)
    nodes_df = pd.read_csv(node_details_path)
    execution_times_df = pd.read_csv(execution_time_path)
    costs_df = pd.read_csv(cost_table_path)
    
    return tasks_df, nodes_df, execution_times_df, costs_df

def task_assignment_algorithm(tasks_df, nodes_df, execution_times_df, costs_df, max_tasks_per_node=None, time_weight=0.6, cost_weight=0.4):
    """
    Assign tasks to nodes to minimize execution time and cost.
    
    Parameters:
    tasks_df: DataFrame from TaskDetails.csv
    nodes_df: DataFrame from NodeDetails.csv
    execution_times_df: DataFrame from ExecutionTable.csv
    costs_df: DataFrame from CostTable.csv
    max_tasks_per_node: Maximum number of tasks that can be assigned to a single node
    time_weight: Weight given to execution time in the score calculation
    cost_weight: Weight given to cost in the score calculation
    
    Returns:
    assignment_df: DataFrame containing task-to-node assignments
    """
    
    node_loads = {node_id: 0 for node_id in nodes_df['NodeID']}
    if max_tasks_per_node is None:
        max_tasks_per_node = float('inf')
    
    
    node_capacities = {node_id: cpu_rate for node_id, cpu_rate in zip(nodes_df['NodeID'], nodes_df['CPU rate (MIPS)'])}
    
    
    remaining_capacity = node_capacities.copy()
    
    
    task_priorities = tasks_df.sort_values(by='Number of instructions (109 instructions)', ascending=False)
    
    
    assignments = []
    
    
    for _, task in task_priorities.iterrows():
        task_id = task['TaskID']
        task_size = task['Number of instructions (109 instructions)']
        
        best_score = float('inf')
        best_node = None
        best_time = None
        best_cost = None
        
        
        for _, node in nodes_df.iterrows():
            node_id = node['NodeID']
            
            
            if node_loads[node_id] >= max_tasks_per_node:
                continue
                
            
            
            if remaining_capacity[node_id] < task_size * 0.1:  
                continue
            
            
            execution_time = execution_times_df.loc[execution_times_df['NodeID'] == node_id, task_id].values[0]
            cost = costs_df.loc[costs_df['NodeID'] == node_id, task_id].values[0]
            
            
            score = (time_weight * execution_time) + (cost_weight * cost)
            
            if score < best_score:
                best_score = score
                best_node = node_id
                best_time = execution_time
                best_cost = cost
        
        if best_node is not None:
            
            assignments.append({
                'TaskID': task_id,
                'NodeID': best_node,
                'ExecutionTime': best_time,
                'Cost': best_cost,
                'Score': best_score
            })
            
            
            node_loads[best_node] += 1
            remaining_capacity[best_node] -= task_size * 0.1  
    
    
    assignment_df = pd.DataFrame(assignments)
    return assignment_df

def calculate_performance_metrics(assignment_df):
    """
    Calculate performance metrics for the assignment.
    
    Parameters:
    assignment_df: DataFrame containing task assignments
    
    Returns:
    metrics: Dictionary of performance metrics
    """
    
    total_execution_time = assignment_df['ExecutionTime'].sum()
    total_cost = assignment_df['Cost'].sum()
    avg_execution_time = assignment_df['ExecutionTime'].mean()
    avg_cost = assignment_df['Cost'].mean()
    
    
    node_execution_times = {}
    for _, assignment in assignment_df.iterrows():
        node_id = assignment['NodeID']
        if node_id not in node_execution_times:
            node_execution_times[node_id] = 0
        node_execution_times[node_id] += assignment['ExecutionTime']
    
    makespan = max(node_execution_times.values()) if node_execution_times else 0
    
    
    tasks_per_node = assignment_df['NodeID'].value_counts().to_dict()
    
    
    all_nodes = set(assignment_df['NodeID'].unique())
    for node in all_nodes:
        if node not in tasks_per_node:
            tasks_per_node[node] = 0
    
    load_balance = np.std(list(tasks_per_node.values()))
    
    
    metrics = {
        'total_execution_time': total_execution_time,
        'total_cost': total_cost,
        'avg_execution_time': avg_execution_time,
        'avg_cost': avg_cost,
        'makespan': makespan,
        'tasks_per_node': tasks_per_node,
        'load_balance': load_balance,
        'node_execution_times': node_execution_times
    }
    
    return metrics

def display_task_assignment(assignment_df):
    """
    Display a clear breakdown of which tasks are assigned to which nodes.
    
    Parameters:
    assignment_df: DataFrame containing task assignments
    """
    
    node_to_tasks = defaultdict(list)
    for _, row in assignment_df.iterrows():
        node_to_tasks[row['NodeID']].append({
            'TaskID': row['TaskID'],
            'ExecutionTime': row['ExecutionTime'],
            'Cost': row['Cost']
        })
    
    
    print("\n===== TASK ASSIGNMENT BREAKDOWN =====")
    for node_id, tasks in node_to_tasks.items():
        print(f"\n{node_id}:")
        print(f"Number of tasks assigned: {len(tasks)}")
        print("Tasks:")
        
        total_node_time = 0
        total_node_cost = 0
        
        
        sorted_tasks = sorted(tasks, key=lambda x: x['ExecutionTime'], reverse=True)
        for task in sorted_tasks:
            print(f"  {task['TaskID']}: Execution Time = {task['ExecutionTime']:.2f}, Cost = {task['Cost']:.2f}")
            total_node_time += task['ExecutionTime']
            total_node_cost += task['Cost']
        
        print(f"Total node execution time: {total_node_time:.2f}")
        print(f"Total node cost: {total_node_cost:.2f}")
    
    
    print("\n===== SUMMARY =====")
    tasks_per_node = assignment_df['NodeID'].value_counts().to_dict()
    for node_id, count in sorted(tasks_per_node.items()):
        print(f"{node_id}: {count} tasks")

def visualize_task_distribution(assignment_df, tasks_df=None):
    """
    Create visualizations of the task distribution across nodes.
    
    Parameters:
    assignment_df: DataFrame containing task assignments
    tasks_df: Optional DataFrame with task details for more detailed visualization
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    
    tasks_per_node = assignment_df['NodeID'].value_counts().sort_index()
    axes[0, 0].bar(tasks_per_node.index, tasks_per_node.values)
    axes[0, 0].set_title('Number of Tasks Assigned to Each Node')
    axes[0, 0].set_xlabel('Node ID')
    axes[0, 0].set_ylabel('Number of Tasks')
    
    
    axes[0, 1].pie(tasks_per_node.values, labels=tasks_per_node.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Task Distribution Across Nodes')
    
    
    node_execution_times = {}
    for _, assignment in assignment_df.iterrows():
        node_id = assignment['NodeID']
        if node_id not in node_execution_times:
            node_execution_times[node_id] = 0
        node_execution_times[node_id] += assignment['ExecutionTime']
    
    sorted_nodes = sorted(node_execution_times.keys())
    execution_times = [node_execution_times[node] for node in sorted_nodes]
    
    axes[1, 0].bar(sorted_nodes, execution_times)
    axes[1, 0].set_title('Total Execution Time per Node')
    axes[1, 0].set_xlabel('Node ID')
    axes[1, 0].set_ylabel('Execution Time')
    
    
    node_costs = {}
    for _, assignment in assignment_df.iterrows():
        node_id = assignment['NodeID']
        if node_id not in node_costs:
            node_costs[node_id] = 0
        node_costs[node_id] += assignment['Cost']
    
    sorted_nodes = sorted(node_costs.keys())
    costs = [node_costs[node] for node in sorted_nodes]
    
    axes[1, 1].bar(sorted_nodes, costs)
    axes[1, 1].set_title('Total Cost per Node')
    axes[1, 1].set_xlabel('Node ID')
    axes[1, 1].set_ylabel('Cost')
    
    plt.tight_layout()
    plt.show()
    
    
    if tasks_df is not None:
        
        merged_df = pd.merge(
            assignment_df, 
            tasks_df[['TaskID', 'Number of instructions (109 instructions)']], 
            on='TaskID'
        )
        
        
        task_size_matrix = pd.pivot_table(
            merged_df, 
            values='Number of instructions (109 instructions)', 
            index='NodeID', 
            columns='TaskID', 
            fill_value=0
        )
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(task_size_matrix, cmap='YlGnBu', annot=True, fmt='.1f')
        plt.title('Task Size Distribution Across Nodes')
        plt.show()



















def run_full_simulation(task_details_path, node_details_path, execution_time_path, cost_table_path, 
                       max_tasks_per_node=None, time_weight=0.6, cost_weight=0.4,
                       display_results=True, visualize=True, export_report=True):
    """
    Run a full simulation of the task assignment algorithm with visualization options.
    
    Parameters:
    task_details_path: Path to TaskDetails.csv
    node_details_path: Path to NodeDetails.csv
    execution_time_path: Path to ExecutionTable.csv
    cost_table_path: Path to CostTable.csv
    max_tasks_per_node: Maximum tasks per node
    time_weight: Weight for execution time in scoring
    cost_weight: Weight for cost in scoring
    display_results: Whether to display detailed assignment results
    visualize: Whether to generate visualizations
    export_report: Whether to export a CSV report
    
    Returns:
    results: Dictionary containing assignments and metrics
    """
    
    tasks_df, nodes_df, execution_times_df, costs_df = load_data(
        task_details_path, node_details_path, execution_time_path, cost_table_path
    )
    
    
    assignments = task_assignment_algorithm(
        tasks_df, nodes_df, execution_times_df, costs_df, 
        max_tasks_per_node, time_weight, cost_weight
    )
    
    
    metrics = calculate_performance_metrics(assignments)
    
    
    if display_results:
        display_task_assignment(assignments)
        
        print("\n===== PERFORMANCE METRICS =====")
        print(f"Total execution time: {metrics['total_execution_time']:.2f}")
        print(f"Total cost: {metrics['total_cost']:.2f}")
        print(f"Makespan: {metrics['makespan']:.2f}")
        print(f"Load balance (std dev of tasks per node): {metrics['load_balance']:.2f}")
    
    
    
    
    
    
    
    
    
    
    results = {
        'assignments': assignments,
        'metrics': metrics
    }
    
    return results

def print_tabular_assignment(assignment_df):
    """
    Print task assignments in a nice tabular format
    
    Parameters:
    assignment_df: DataFrame containing task assignments
    """
    
    node_task_mapping = defaultdict(list)
    
    for _, row in assignment_df.iterrows():
        node_task_mapping[row['NodeID']].append(row['TaskID'])
    
    
    max_tasks = max(len(tasks) for tasks in node_task_mapping.values())
    
    
    print("\n===== TASK-TO-NODE ASSIGNMENT TABLE =====")
    header = "Node ID | Tasks Assigned"
    print(header)
    print("-" * len(header) + "--------" * max_tasks)
    
    
    for node_id in sorted(node_task_mapping.keys()):
        tasks = node_task_mapping[node_id]
        task_str = ", ".join(tasks)
        print(f"{node_id:7} | {task_str}")


if __name__ == "__main__":
    
    task_details_path = "TaskDetails.csv"
    node_details_path = "NodeDetails.csv"
    execution_time_path = "ExecutionTable.csv"
    cost_table_path = "CostTable.csv"
    
    
    print("Running task assignment algorithm...")
    results = run_full_simulation(
        task_details_path, node_details_path, execution_time_path, cost_table_path,
        max_tasks_per_node=None, time_weight=0.6, cost_weight=0.4,
        display_results=True, visualize=True, export_report=True
    )
    
    
    print_tabular_assignment(results['assignments'])
