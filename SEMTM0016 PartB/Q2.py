import numpy as np
import copy
import time
from envs.simple_dungeonworld_env import DungeonMazeEnv
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt

dim = 8
maze_dim = dim - 2  
gamma = 1
eps = 1e-10
reward = -1
num_directions = 4  
num_actions = 3  
initial_value = np.zeros((dim, dim, num_directions))


def extract_maze_grid(env):
    maze_array = np.zeros((env.grid_size, env.grid_size), dtype=int)
    for cell in env.maze.grid:
        if cell is not None and cell.type == "wall":
            x, y = cell.pos  
            maze_array[y, x] = 1
    return maze_array

def next_state(i, j, current_d, action, maze_grid):
    num = maze_grid.shape[0]
    next_d = current_d

    if action == 0:  
        next_d = (next_d - 1) % 4
    elif action == 1:  
        next_d = (next_d + 1) % 4

    if next_d == 0:  
        next_i, next_j = max(i-1, 0), j
    elif next_d == 1:  
        next_i, next_j = i, min(j+1, num-1)
    elif next_d == 2:  
        next_i, next_j = min(i+1, num-1), j
    else:  
        next_i, next_j = i, max(j-1, 0)

    if maze_grid[next_i, next_j] == 1:
        return i, j, next_d
    else:
        return next_i, next_j, next_d

def get_optimal(q_matrix, maze_grid):
    num = q_matrix.shape[0]
    policy = np.ones([num, num, num_directions, num_actions]) / num_actions
    for i in range(num):
        for j in range(num):
            if maze_grid[i, j] == 1:
                continue
            if (i == num-2 and j == num-2):
                continue
            for d in range(num_directions):
                q_value = q_matrix[i, j, d].copy()
                max_q = np.max(q_value)
                mask = (q_value == max_q)
                count_max = np.sum(mask)
                policy_state = np.zeros(num_actions)
                policy_state[mask] = 1.0 / count_max
                policy[i, j, d] = policy_state
    return policy

def get_q_matrix(value_matrix, maze_grid):
    num = value_matrix.shape[0]
    q_matrix = np.zeros([num, num, num_directions, num_actions])
    for i in range(num):
        for j in range(num):
            if maze_grid[i, j] == 1:
                q_matrix[i, j, :, :] = -100
                continue
            if (i == num-2 and j == num-2):
                q_matrix[i, j, :, :] = 0
                continue
            for d in range(num_directions):
                for action in range(num_actions):
                    next_i, next_j, next_d = next_state(i, j, d, action, maze_grid)
                    intended_i, intended_j = next_i, next_j
                    if action == 2:  
                        if next_d == 0: intended_i = i-1
                        elif next_d == 1: intended_j = j+1
                        elif next_d == 2: intended_i = i+1
                        else: intended_j = j-1
                    if action == 2 and maze_grid[intended_i, intended_j] == 1:
                        q_matrix[i, j, d, action] = -100
                    else:
                        if (next_i == num-2 and next_j == num-2):
                            next_value = 0
                        else:
                            next_value = value_matrix[next_i, next_j, next_d]
                        q_matrix[i, j, d, action] = reward + gamma * next_value
    return q_matrix

def evaluate_policy(policy, eps, maze_grid):
    num = policy.shape[0]
    # value_matrix = np.zeros((num, num, num_directions))
    # value_matrix = np.random.randn(num, num, num_directions)
    # value_matrix = np.ones((num, num, num_directions))*-100
    value_matrix = initial_value.copy()
    evaluate_policy_iteration = 0
    value_mse_list = []  

    value_matrix[-2, -2, :] = 0  
    for i in range(num):
        for j in range(num):
            if maze_grid[i, j] == 1:
                value_matrix[i, j, :] = -100

    while True:
        value_prev = copy.deepcopy(value_matrix)
        for i in range(num):
            for j in range(num):
                if maze_grid[i, j] == 1 or (i == num-2 and j == num-2):
                    continue
                for d in range(num_directions):
                    value = 0
                    for action in range(num_actions):
                        next_i, next_j, next_d = next_state(i, j, d, action, maze_grid)
                        intended_i, intended_j = next_i, next_j
                        if action == 2:  
                            if next_d == 0: intended_i = i-1
                            elif next_d == 1: intended_j = j+1
                            elif next_d == 2: intended_i = i+1
                            else: intended_j = j-1
                        if action == 2 and maze_grid[intended_i, intended_j] == 1:
                            action_reward = -100
                            next_value = -100  
                        else:
                            action_reward = -1
                            if (next_i == num-2 and next_j == num-2):
                                next_value = 0
                            else:
                                next_value = value_prev[next_i, next_j, next_d]
                        value += policy[i, j, d, action] * (action_reward + gamma * next_value)
                    value_matrix[i, j, d] = value
        evaluate_policy_iteration += 1
        diff = np.mean((value_prev - value_matrix)**2) 
        value_mse_list.append(diff)       
        if diff < eps:
            break
    return value_matrix, evaluate_policy_iteration, value_mse_list

def policy_iteration(dim, eps, maze_grid):
    policy = np.ones([dim, dim, num_directions, num_actions]) / num_actions
    iteration_count = 0
    value_mse_list = []
    while True:
        prev_policy = copy.deepcopy(policy)
        value_matrix, evaluate_policy_iteration, mse_list = evaluate_policy(policy, eps, maze_grid)
        value_mse_list.extend(mse_list)

        q_matrix = get_q_matrix(value_matrix, maze_grid)
        policy = get_optimal(q_matrix, maze_grid)
        diff = np.mean((prev_policy-policy)**2)
        iteration_count += 1
        
        if diff < eps:
            break
    return policy, value_matrix, iteration_count, value_mse_list


def visualise(policy, caption, direction=2): 
    plt.figure(figsize=(dim, dim)) 
    ax = plt.gca() 
    ax.xaxis.set_major_locator(MultipleLocator(1)) 
    ax.yaxis.set_major_locator(MultipleLocator(1)) 
    plt.xlim(0, dim) 
    plt.ylim(0, dim) 
     
    for i in range(1, maze_dim + 1): 
        for j in range(1, maze_dim + 1): 
            if (i == dim-2 and j == dim-2): 
                continue 
            for action, action_prob in enumerate(policy[i, j, direction]): 
                if action_prob > 0: 
                    offsets = [(0.3, 0), (-0.3, 0), (0, -0.3)] 
                    plt.arrow(j+0.5, dim-i-0.5, *offsets[action], head_width=0.05, head_length=0.1, fc='k', ec='k') 
    plt.grid(True) 
    plt.title(caption)
    plt.show() 


def update_optimal_value_function(optimal_value_matrix, maze_grid):
    value_matrix = copy.deepcopy(optimal_value_matrix)
    num = len(optimal_value_matrix)
    for i in range(num):
        for j in range(num):
            if maze_grid[i, j] == 1 or (i == num-2 and j == num-2):
                continue  
            for d in range(num_directions):
                max_value = -np.inf
                for action in range(num_actions):
                    next_i, next_j, next_d = next_state(i, j, d, action, maze_grid)
                    intended_i, intended_j = next_i, next_j
                    if action == 2:  
                        if next_d == 0: intended_i = i-1
                        elif next_d == 1: intended_j = j+1
                        elif next_d == 2: intended_i = i+1
                        else: intended_j = j-1
                    if action == 2 and maze_grid[intended_i, intended_j] == 1:
                        current_value = -100  
                    else:
                        if (next_i == num-2 and next_j == num-2):
                            next_value = 0
                        else:
                            next_value = value_matrix[next_i, next_j, next_d]
                        current_value = reward + gamma * next_value
                    max_value = max(max_value, current_value)
                optimal_value_matrix[i, j, d] = max_value
    return optimal_value_matrix

def value_iteration(dim, eps, maze_grid):
    # optimal_value_matrix = np.zeros((dim, dim, num_directions))
    # optimal_value_matrix = np.random.randn(dim, dim, num_directions)
    # optimal_value_matrix = np.ones((dim, dim, num_directions))*-100
    optimal_value_matrix = initial_value.copy()
    optimal_value_matrix[-2, -2, :] = 0
    iteration_count = 0
    value_mse_list = []
    for i in range(dim):
        for j in range(dim):
            if maze_grid[i, j] == 1:
                optimal_value_matrix[i, j, :] = -100
    while True:
        prev_optimal_value_matrix = copy.deepcopy(optimal_value_matrix)
        optimal_value_matrix = update_optimal_value_function(optimal_value_matrix, maze_grid)
        diff = np.mean((prev_optimal_value_matrix - optimal_value_matrix)**2)
        iteration_count += 1
        value_mse_list.append(diff)
        if diff < eps:
            break
    return optimal_value_matrix, iteration_count, value_mse_list

def compute_mse(matrix1, matrix2):
    return np.mean((matrix1 - matrix2) ** 2)

def plot_stability(policy_mse_list, value_mse_list):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(policy_mse_list) + 1), policy_mse_list, label='Policy Iteration (Policy Evaluation)')
    plt.plot(range(1, len(value_mse_list) + 1), value_mse_list, label='Value Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('MSE of Value Function')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


env = DungeonMazeEnv(render_mode=None, grid_size=dim)
obs, info = env.reset(seed=124)
maze_grid = extract_maze_grid(env)
    
policy, value_matrix, p_iteration_count, policy_mse_list = policy_iteration(dim, eps, maze_grid)
caption1 = "Policy Iteration"
visualise(policy, caption1)
    

optimal_value_matrix, v_iteration_count, value_mse_list = value_iteration(dim, eps, maze_grid)
optimal_q_matrix = get_q_matrix(optimal_value_matrix, maze_grid)
optimal_policy = get_optimal(optimal_q_matrix, maze_grid)
caption2 = "Value Iteration"
visualise(optimal_policy, caption2)
    

plot_stability(policy_mse_list, value_mse_list)
# plot_stability(diff_value_inpolicy, value_mse_list)


for h in range(3):
    if h == 0:
        initial_value = np.ones((dim, dim, num_directions))*-100
    if h == 1:
        initial_value = np.zeros((dim, dim, num_directions))
    if h == 2:
        initial_value = np.random.randn(dim, dim, num_directions)

    env = DungeonMazeEnv(render_mode=None, grid_size=dim)
    obs, info = env.reset(seed=124)
    maze_grid = extract_maze_grid(env)
    
    start_time = time.time()
    policy, value_matrix, p_iteration_count, policy_mse_list = policy_iteration(dim, eps, maze_grid)    
    end_time = time.time()
    policy_iteration_time = end_time - start_time

    start_time = time.time()
    optimal_value_matrix, v_iteration_count, value_mse_list = value_iteration(dim, eps, maze_grid)
    optimal_q_matrix = get_q_matrix(optimal_value_matrix, maze_grid)
    optimal_policy = get_optimal(optimal_q_matrix, maze_grid)
    end_time = time.time()
    value_iteration_time = end_time - start_time


    if h == 0:
        p_iteration_count_negaini = p_iteration_count
        policy_iteration_time_negaini = policy_iteration_time
        v_iteration_count_negaini = v_iteration_count
        value_iteration_time_negaini = value_iteration_time
    if h == 1:
        p_iteration_count_zeroini = p_iteration_count
        policy_iteration_time_zeroini = policy_iteration_time
        v_iteration_count_zeroini = v_iteration_count
        value_iteration_time_zeroini = value_iteration_time
    if h == 2:
        p_iteration_count_ranini = p_iteration_count
        policy_iteration_time_ranini = policy_iteration_time
        v_iteration_count_ranini = v_iteration_count
        value_iteration_time_ranini = value_iteration_time

print("iteration under negative initial value")
print(f"policy iteration: {p_iteration_count_negaini}")
print(f"policy iteration time: {policy_iteration_time_negaini}")
print(f"value iteration: {v_iteration_count_negaini}")
print(f"value iteration time: {value_iteration_time_negaini}")

print("iteration under zero initial value")
print(f"policy iteration: {p_iteration_count_zeroini}")
print(f"policy iteration time: {policy_iteration_time_zeroini}")
print(f"value iteration: {v_iteration_count_zeroini}")
print(f"value iteration time: {value_iteration_time_zeroini}")

print("iteration under random initial value")
print(f"policy iteration: {p_iteration_count_ranini}")
print(f"policy iteration time: {policy_iteration_time_ranini}")
print(f"value iteration: {v_iteration_count_ranini}")
print(f"value iteration time: {value_iteration_time_ranini}")




