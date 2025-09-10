#%%
import numpy as np
import random
from envs.simple_dungeonworld_env import DungeonMazeEnv
import matplotlib.pyplot as plt

SIZE = 10
gamma = 0.99  # Discount factor
max_steps = 100  # Maximum number of steps in each trajectory
num_simulations = 1000 # Number of trajectories to sample
alpha = 0.1  
epislon = 1e-2

env = DungeonMazeEnv(render_mode=None, grid_size=SIZE)

def random_policy(state):
    return random.choice([0, 1, 2])  # Choose action randomly

# Rollout function for sampling trajectories
def rollout(env, policy, num_simulations, max_steps=100):
    trajs = []  # To store the trajectory
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

def MC_value_estimation(policy):
    trajs = rollout(env, policy, num_simulations)
    returns = {}  
    value_table = np.zeros((SIZE, SIZE, 4))  # Value table for each state (x, y, direction)
    previous_value_table = np.copy(value_table)
    Gs = []
    mse_list = []

    for traj in trajs:
        states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
        visited = set()
        G = 0  
        
        for t in reversed(range(len(states)-1)):  
            state, action, reward = states[t], actions[t], rewards[t]
            G = reward + gamma * G  
            
            x, y = state["robot_position"]
            d = state["robot_direction"]

            if (x, y, d) not in visited:
                visited.add((x, y, d))
                if (x, y, d) not in returns:
                    returns[(x, y, d)] = []
                returns[(x, y, d)].append(G)
                value_table[x, y, d] = np.mean(returns[(x, y, d)])
        
        mse_list.append(compute_mse(value_table, previous_value_table))
        previous_value_table = np.copy(value_table)  
        Gs.append(G)  

    return value_table, Gs, mse_list


def TD_value_estimation(policy):
    value_table = np.zeros((SIZE, SIZE, 4))  # Value table for grid and 4 directions
    trajs = rollout(env, policy, num_simulations)  # Sample trajectories
    Gs = []  
    mse_list = []
    
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
        
        mse_list.append(compute_mse(value_table, previous_value_table))
        if compute_mse(previous_value_table, value_table) < epislon:
            break

        Gs.append(np.sum(rewards))  
    
    return value_table, Gs, mse_list

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


mc_value_table, mc_reward, mc_mse = MC_value_estimation(random_policy)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mc_reward, label="Monte Carlo")
plt.xlabel('Trajectory')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward (MC)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mc_mse, label="Monte Carlo")
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('MSE Convergence (MC)')
plt.legend()

plt.tight_layout()
plt.show()


td_value_table, td_reward, td_mse = TD_value_estimation(random_policy)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(td_reward, label="Temporal Difference")
plt.xlabel('Trajectory')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward (TD)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(td_mse, label="Temporal Difference")
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('MSE Convergence (TD)')
plt.legend()

plt.tight_layout()
plt.show()



title1 = "Value Table Visualization for MC"
title2 = "Value Table Visualization for TD"
plot_value_table(mc_value_table, title1)
plot_value_table(td_value_table, title2)

#%%
