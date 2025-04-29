import pandas as pd


task_df = pd.read_csv('TaskDetails.csv')
node_df = pd.read_csv('NodeDetails.csv')
execution_df = pd.read_csv('ExecutionTable.csv', index_col='NodeID')
cost_df = pd.read_csv('CostTable.csv', index_col='NodeID')

def efficient_task_assignment(task_df, node_df, execution_df, cost_df, time_weight=0.7, cost_weight=0.3):
    """
    Assigns tasks to nodes to minimize a weighted sum of execution time and cost.

    Args:
        task_df (pd.DataFrame): DataFrame containing task details.
        node_df (pd.DataFrame): DataFrame containing node details.
        execution_df (pd.DataFrame): DataFrame containing pre-calculated execution times.
        cost_df (pd.DataFrame): DataFrame containing pre-calculated costs.
        time_weight (float): Weight for the execution time in the evaluation function.
        cost_weight (float): Weight for the execution cost in the evaluation function.

    Returns:
        dict: A dictionary mapping TaskID to the assigned NodeID.
    """

    unassigned_tasks = list(task_df['TaskID'])
    available_nodes = list(node_df['NodeID'])
    assignment = {}
    node_current_load = {node: [] for node in available_nodes}  

    while unassigned_tasks:
        best_assignment = None
        min_weighted_score = float('inf')

        for task_id in unassigned_tasks:
            for node_id in available_nodes:
                
                execution_time = execution_df.loc[node_id, task_id]
                cost = cost_df.loc[node_id, task_id]

                
                weighted_score = (time_weight * execution_time) + (cost_weight * cost)

                if weighted_score < min_weighted_score:
                    min_weighted_score = weighted_score
                    best_assignment = {'task': task_id, 'node': node_id}

        if best_assignment:
            task_to_assign = best_assignment['task']
            node_to_assign = best_assignment['node']
            assignment[task_to_assign] = node_to_assign
            node_current_load[node_to_assign].append(task_to_assign)
            unassigned_tasks.remove(task_to_assign)
        else:
            print("Warning: Could not assign all tasks.")
            break

    return assignment

if __name__ == "__main__":
    
    optimal_assignment = efficient_task_assignment(task_df, node_df, execution_df, cost_df)
    print("Optimal Task Assignment:")
    for task, node in optimal_assignment.items():
        print(f"Task: {task} -> Node: {node}")

    
    total_estimated_time = 0
    total_estimated_cost = 0
    for task, node in optimal_assignment.items():
        total_estimated_time += execution_df.loc[node, task]
        total_estimated_cost += cost_df.loc[node, task]

    print(f"\nTotal Estimated Execution Time: {total_estimated_time:.4f}")
    print(f"Total Estimated Cost: {total_estimated_cost:.4f}")
