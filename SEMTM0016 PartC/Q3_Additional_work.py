#%%
# Improved Q-learning
from envs.simple_dungeonworld_env import DungeonMazeEnv, Directions, Actions
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math

seed = 124
dim = 14  # Full grid size
gamma = 0.99  # Discount factor

env = DungeonMazeEnv(render_mode=None, grid_size=dim)
num_episodes = 2000
learning_rate = 0.5
epsilon = 1e-2
num_replicates = 10

def epsilon_greedy(epsilon, q_table, state):
    if np.random.rand() < epsilon:
        return np.random.choice(3)  
    else:
        return np.argmax(q_table[state])  
    
def q_learning():
    q_learning_rewards = []
    steps_per_episode = []
    q_table_history = []  
    q_table = np.zeros((dim, dim, 4, 3))  

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)  
        cumulated_reward = 0
        step_count = 0
        done = False

        while not done:
            x, y = state["robot_position"]
            direction = state["robot_direction"] 

            action = epsilon_greedy(epsilon, q_table, (x, y, direction))

            next_state, reward, done, _, _ = env.step(action)

            x_next, y_next = next_state["robot_position"]
            direction_next = next_state["robot_direction"]
            if reward == -1:
                reward = reward_function(next_state["robot_position"], [dim-2, dim-2])

            if np.array_equal(next_state["robot_position"], [dim-2, dim-2]):
                    print("reach goal")

            q_table[x, y, direction, action] += learning_rate * (
                reward + gamma * np.max(q_table[x_next, y_next, direction_next]) - q_table[x, y, direction, action]
            )

            cumulated_reward += reward
            state = next_state

            step_count += 1
            if step_count >= 100:  
                done = True

        q_learning_rewards.append(cumulated_reward)
        steps_per_episode.append(step_count)
        q_table_history.append(np.copy(q_table))

        print(f"Episode {episode+1}/{num_episodes} - Cumulative Reward: {cumulated_reward}")

    return q_learning_rewards, steps_per_episode, q_table_history

def run_q_learning(num_replicates):
    all_q_learning_rewards = []
    all_q_learning_steps = []
    all_q_learning_q_table_history = []

    for _ in range(num_replicates):
        q_learning_rewards, q_learning_steps, q_learning_q_table_history = q_learning()
        all_q_learning_rewards.append(q_learning_rewards)
        all_q_learning_steps.append(q_learning_steps)
        all_q_learning_q_table_history.append(q_learning_q_table_history)

    # Averaging the results across all replicates
    q_learning_rewards_avg = np.mean(all_q_learning_rewards, axis=0)
    q_learning_steps_avg = np.mean(all_q_learning_steps, axis=0)
    q_learning_q_table_history_avg = np.mean(all_q_learning_q_table_history, axis=0)

    return q_learning_rewards_avg, q_learning_steps_avg, q_learning_q_table_history_avg


def reward_function(current_position, target_position):

    current_distance = math.sqrt((current_position[0] - target_position[0])**2 + (current_position[1] - target_position[1])**2)
    
    max_distance = math.sqrt((1 - (dim-2))**2 + (1 - (dim-2))**2)
    
    # Reward based on distance
    distance_reward = 1 - (current_distance / max_distance)
    penalty = -1    
    return penalty + distance_reward

q_learning_improved_rewards_avg, q_learning_improved_steps_avg, q_learning_improved_q_table_history_avg = run_q_learning(num_replicates)


#%%
# Original Q-learning

def q_learning():
    q_learning_rewards = []
    steps_per_episode = []
    q_table_history = []  
    q_table = np.zeros((dim, dim, 4, 3)) 

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)  
        cumulated_reward = 0
        step_count = 0
        done = False

        while not done:
            x, y = state["robot_position"]
            direction = state["robot_direction"]  

            action = epsilon_greedy(epsilon, q_table, (x, y, direction))

            next_state, reward, done, _, _ = env.step(action)

            x_next, y_next = next_state["robot_position"]
            direction_next = next_state["robot_direction"]

            if np.array_equal(next_state["robot_position"], [dim-2, dim-2]):
                print("reach goal")

            q_table[x, y, direction, action] += learning_rate * (
                reward + gamma * np.max(q_table[x_next, y_next, direction_next]) - q_table[x, y, direction, action]
            )

            cumulated_reward += reward
            state = next_state

            step_count += 1
            if step_count >= 100:  # Limit to 100 steps
                done = True

        q_learning_rewards.append(cumulated_reward)
        steps_per_episode.append(step_count)
        q_table_history.append(np.copy(q_table))

    return q_learning_rewards, steps_per_episode, q_table_history

q_learning_rewards_avg, q_learning_steps_avg, q_learning_q_table_history_avg = run_q_learning(num_replicates)

plt.plot(q_learning_improved_rewards_avg, label="Q-Learning-Improved")
plt.plot(q_learning_rewards_avg, label="Q-Learning")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.show()

#%%
def plot_steps_per_episode(q_learning_steps_avg, q_learning_improved_steps_avg):
    plt.plot(q_learning_steps_avg, label="Q-Learning", color='blue')
    plt.plot(q_learning_improved_steps_avg, label="Q-Learning-Improved", color='green')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Average Steps per Episode (Convergence Time)')
    plt.legend()
    plt.show()

# 2. Q-Table Convergence (Track Q-value changes)
def plot_q_table_convergence(q_learning_q_table_history_avg, q_learning_improved_q_table_history_avg):
    q_learning_q_table_changes_avg = [np.max(np.abs(q_learning_q_table_history_avg[i] - q_learning_q_table_history_avg[i-1])) for i in range(1, len(q_learning_q_table_history_avg))]
    q_learning_improved_q_table_changes_avg = [np.max(np.abs(q_learning_improved_q_table_history_avg[i] - q_learning_improved_q_table_history_avg[i-1])) for i in range(1, len(q_learning_improved_q_table_history_avg))]
    
    plt.plot(q_learning_q_table_changes_avg, label="Q-Learning", color='blue')
    plt.plot(q_learning_improved_q_table_changes_avg, label="Q-Learning-Improved", color='green')
    plt.xlabel('Episode')
    plt.ylabel('Q-value Change')
    plt.title('Q-Table Convergence')
    plt.legend()
    plt.show()

# 3. Reward Per Step (Efficiency)
def plot_reward_per_step(q_learning_rewards_avg, q_learning_steps_avg, q_learning_improved_rewards_avg, q_learning_improved_steps_avg):
    q_learning_rewards_per_step_avg = [q_learning_rewards_avg[i] / q_learning_steps_avg[i] for i in range(len(q_learning_rewards_avg))]
    q_learning_improved_rewards_per_step_avg = [q_learning_improved_rewards_avg[i] / q_learning_improved_steps_avg[i] for i in range(len(q_learning_improved_rewards_avg))]
    
    plt.plot(q_learning_rewards_per_step_avg, label="Q-Learning", color='blue')
    plt.plot(q_learning_improved_rewards_per_step_avg, label="Q-Learning-Improved", color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward Per Step')
    plt.title('Average Reward Per Step (Efficiency)')
    plt.legend()
    plt.show()

# 4. Q-Value Distribution (Track the distribution of Q-values at the final episode)
def plot_q_value_distribution(q_learning_q_table_avg, q_learning_improved_q_table_avg, episode):
    # Flatten and plot the Q-values for both Q-learning and SARSA at the final episode
    all_q_values_q_learning = q_learning_q_table_avg.flatten()
    all_q_values_q_learning_improved = q_learning_improved_q_table_avg.flatten()
    
    plt.hist(all_q_values_q_learning, bins=50, alpha=0.5, label='Q-Learning', color='blue')
    plt.hist(all_q_values_q_learning_improved, bins=50, alpha=0.5, label='Q-Learning-Improved', color='green')
    plt.xlabel('Q-Value')
    plt.ylabel('Frequency')
    plt.title(f'Q-Value Distribution at Episode {episode}')
    plt.legend()
    plt.show()

    
plot_steps_per_episode(q_learning_steps_avg, q_learning_improved_steps_avg)

# Plot Q-table Convergence
plot_q_table_convergence(q_learning_q_table_history_avg, q_learning_improved_q_table_history_avg)

# Plot Reward per Step (Efficiency)
plot_reward_per_step(q_learning_rewards_avg, q_learning_steps_avg, q_learning_improved_rewards_avg, q_learning_improved_steps_avg)

# Plot Q-Value Distribution (choose a specific episode to plot)
plot_q_value_distribution(q_learning_q_table_history_avg[-1], q_learning_improved_q_table_history_avg[-1], num_episodes)

# %%
