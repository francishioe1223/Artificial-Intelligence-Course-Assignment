# %%
import numpy as np
import random
from envs.simple_dungeonworld_env import DungeonMazeEnv
import matplotlib.pyplot as plt

# Define the environment size and hyperparameters
SIZE = 10
gamma = 0.99  # Discount factor
max_steps = 100  # Maximum number of steps in each trajectory
num_simulations = 500 # Number of trajectories to sample
alpha = 0.1  
epsilon = 1e-2

env = DungeonMazeEnv(render_mode=None, grid_size=SIZE)

def random_policy(state):
    return random.choice([0, 1, 2])  # Choose action randomly

# Rollout function for sampling trajectories
def rollout(env, policy, num_simulations, max_steps=100):
    trajs = []  
    for _ in range(num_simulations):
        rsum = []
        states = []
        actions = []
        state, _ = env.reset(seed=124)
        
        for _ in range(max_steps):
            action = policy(state)  
            next_state, reward, terminated, _, _ = env.step(action)  
            rsum.append(reward)
            actions.append(action)
            states.append(state)
            state = next_state
            
            if terminated:
                break
        
        states.append(next_state)
        trajs.append({'states': states, 'actions': actions, 'rewards': rsum})  
    return trajs

def compute_mse(value_table, final_value_table):
    return np.mean((value_table - final_value_table) ** 2)

def TD_value_estimation(policy):
    value_table = np.zeros((SIZE, SIZE, 4))  # Value table for grid and 4 directions
    trajs = rollout(env, policy, num_simulations)  
    mse_list_td = []
    Gs_td = []
    
    for traj in trajs:
        states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
        
        for t in range(len(states) - 1):  
            previous_value_table = np.copy(value_table)
            state, action, reward = states[t], actions[t], rewards[t]
            next_state = states[t + 1]
            
            x, y = state["robot_position"]
            d = state["robot_direction"]
            next_x, next_y = next_state["robot_position"]
            next_d = next_state["robot_direction"]
            
            value_table[x, y, d] += alpha * (reward + gamma * value_table[next_x, next_y, next_d] - value_table[x, y, d])
        
        mse_list_td.append(compute_mse(value_table, previous_value_table))
        if compute_mse(previous_value_table, value_table) < epsilon:
            break
        
        Gs_td.append(np.sum(rewards)) 
    
    return value_table, Gs_td, mse_list_td

def MC_value_estimation(policy):
    value_table = np.zeros((SIZE, SIZE, 4))
    trajs = rollout(env, policy, num_simulations)
    Gs_mc = []
    mse_list_mc = []
    
    for traj in trajs:
        states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
        G = 0
        previous_value_table = np.copy(value_table)
        
        # Monte Carlo update
        for t in reversed(range(len(states)-1)):
            G = rewards[t] + gamma * G 
            x, y = states[t]["robot_position"]
            d = states[t]["robot_direction"]
            
            # MC update: V(S_t) <- mean(G_t)
            value_table[x, y, d] += (G - value_table[x, y, d]) / (t + 1)  
        
        mse_list_mc.append(compute_mse(value_table, previous_value_table))
        Gs_mc.append(np.sum(rewards))
    
    return value_table, Gs_mc, mse_list_mc

def plot_results(mse_list_td, mse_list_mc, Gs_td, Gs_mc):
    iterations_td = range(len(mse_list_td))
    iterations_mc = range(len(mse_list_mc))
    
    plt.figure(figsize=(12, 5))
    # MSE Plot
    plt.subplot(1, 2, 1)
    plt.plot(iterations_td, mse_list_td, label='TD(0) MSE', color='b')
    plt.xlabel('Iterations', fontsize = 12)
    plt.ylabel('Mean Squared Error', fontsize = 12)
    plt.yscale('log')
    plt.legend(fontsize = 12)
    plt.subplot(1, 2, 2)
    plt.plot(iterations_mc, mse_list_mc, label='MC MSE', color='r')
    plt.xlabel('Iterations', fontsize = 12)
    plt.ylabel('Mean Squared Error', fontsize = 12)
    plt.yscale('log')
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.show()

    plt.plot(iterations_td, Gs_td, label='TD(0) Return', color='b')
    plt.plot(iterations_mc, Gs_mc, label='MC Return', color='r')
    plt.xlabel('Iterations')
    plt.ylabel('Total Return (G)')
    plt.title('Total Return Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_value_table(value_table, title):
    
    directions = ["Up", "Right", "Down", "Left"]
    plt.figure(figsize=(16, 12))

    for d in range(4):
        plt.subplot(2, 2, d+1)
        plt.imshow(np.transpose(value_table[:, :, d]), cmap='viridis', origin='upper', interpolation='nearest')
        plt.title(f'Value for direction: {directions[d]}', fontsize=14)
        plt.colorbar(label="Value")
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        plt.grid(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

value_td, Gs_td, mse_td = TD_value_estimation(random_policy)
value_mc, Gs_mc, mse_mc = MC_value_estimation(random_policy)

plot_results(mse_td, mse_mc, Gs_td, Gs_mc)

title1 = "Value Table Visualization for MC"
title2 = "Value Table Visualization for TD"
plot_value_table(value_mc, title1)
plot_value_table(value_td, title2)

# %%





