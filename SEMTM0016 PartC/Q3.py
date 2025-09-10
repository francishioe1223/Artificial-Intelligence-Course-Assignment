#%%
from envs.simple_dungeonworld_env import DungeonMazeEnv, Directions, Actions
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

seed = 124
dim = 10  
gamma = 0.99  

env = DungeonMazeEnv(render_mode=None, grid_size=dim)
num_episodes = 1000
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

            q_table[x, y, direction, action] += learning_rate * (
                reward + gamma * np.max(q_table[x_next, y_next, direction_next]) - q_table[x, y, direction, action]
            )

            cumulated_reward += reward
            state = next_state

            step_count += 1
            if step_count >= 100:  
                done = True
                print(f"Episode {episode+1} terminated after 100 steps.")

        q_learning_rewards.append(cumulated_reward)
        steps_per_episode.append(step_count)
        q_table_history.append(np.copy(q_table))


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

    q_learning_rewards_avg = np.mean(all_q_learning_rewards, axis=0)
    q_learning_steps_avg = np.mean(all_q_learning_steps, axis=0)
    q_learning_q_table_history_avg = np.mean(all_q_learning_q_table_history, axis=0)

    return q_learning_rewards_avg, q_learning_steps_avg, q_learning_q_table_history_avg

q_learning_rewards_avg, q_learning_steps_avg, q_learning_q_table_history_avg = run_q_learning(num_replicates)

# %%
# Sarsa

def epsilon_greedy_sarsa(epsilon, q_table, x, y, direction):
    if np.random.rand() < epsilon:
        return np.random.choice(3) 
    else:
        return np.argmax(q_table[x, y, direction])  

def sarsa():
    sarsa_rewards = []
    steps_per_episode = []
    q_table_history = []
    
    q_table = np.zeros((dim, dim, 4, 3) ) 

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)  
        x, y = state["robot_position"]
        direction = state["robot_direction"]
        
        action = epsilon_greedy_sarsa(epsilon, q_table, x, y, direction)
        
        cumulated_reward = 0
        done = False
        step_count = 0
        
        while not done:
            next_state, reward, terminated, _, _ = env.step(action)
            x_next, y_next = next_state["robot_position"]
            direction_next = next_state["robot_direction"]

            action_next = epsilon_greedy_sarsa(epsilon, q_table, x_next, y_next, direction_next)

            q_table[x, y, direction, action] += learning_rate * (
                reward + gamma * q_table[x_next, y_next, direction_next, action_next] - q_table[x, y, direction, action]
            )

            cumulated_reward += reward
            x, y, direction, action = x_next, y_next, direction_next, action_next  
            done = bool(terminated)

            step_count += 1
            if step_count >= 100:  
                done = True
                print(f"Episode {episode+1} terminated after 100 steps.")

        sarsa_rewards.append(cumulated_reward)  
        steps_per_episode.append(step_count)
        q_table_history.append(np.copy(q_table))

        print(f"Episode {episode+1}/{num_episodes} - Cumulative Reward: {cumulated_reward}")

    return sarsa_rewards, steps_per_episode, q_table_history

def run_sarsa(num_replicates):
    all_sarsa_rewards = []
    all_sarsa_steps = []
    all_sarsa_q_table_history = []

    for _ in range(num_replicates):
        sarsa_rewards, sarsa_steps, sarsa_q_table_history = sarsa()
        all_sarsa_rewards.append(sarsa_rewards)
        all_sarsa_steps.append(sarsa_steps)
        all_sarsa_q_table_history.append(sarsa_q_table_history)

    sarsa_rewards_avg = np.mean(all_sarsa_rewards, axis=0)
    sarsa_steps_avg = np.mean(all_sarsa_steps, axis=0)
    sarsa_q_table_history_avg = np.mean(all_sarsa_q_table_history, axis=0)

    return sarsa_rewards_avg, sarsa_steps_avg, sarsa_q_table_history_avg

sarsa_rewards_avg, sarsa_steps_avg, sarsa_q_table_history_avg = run_sarsa(num_replicates)

#%%
def plot_steps_per_episode(q_learning_steps_avg, sarsa_steps_avg):
    plt.plot(q_learning_steps_avg, label="Q-Learning", color='blue')
    plt.plot(sarsa_steps_avg, label="SARSA", color='green')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Average Steps per Episode (Convergence Time)')
    plt.legend()
    plt.show()

def plot_q_table_convergence(q_learning_q_table_history_avg, sarsa_q_table_history_avg):
    q_learning_q_table_changes_avg = [np.max(np.abs(q_learning_q_table_history_avg[i] - q_learning_q_table_history_avg[i-1])) for i in range(1, len(q_learning_q_table_history_avg))]
    sarsa_q_table_changes_avg = [np.max(np.abs(sarsa_q_table_history_avg[i] - sarsa_q_table_history_avg[i-1])) for i in range(1, len(sarsa_q_table_history_avg))]
    
    plt.plot(q_learning_q_table_changes_avg, label="Q-Learning", color='blue')
    plt.plot(sarsa_q_table_changes_avg, label="SARSA", color='green')
    plt.xlabel('Episode')
    plt.ylabel('Q-value Change')
    plt.title('Q-Table Convergence')
    plt.legend()
    plt.show()

def plot_reward_per_step(q_learning_rewards_avg, q_learning_steps_avg, sarsa_rewards_avg, sarsa_steps_avg):
    q_learning_rewards_per_step_avg = [q_learning_rewards_avg[i] / q_learning_steps_avg[i] for i in range(len(q_learning_rewards_avg))]
    sarsa_rewards_per_step_avg = [sarsa_rewards_avg[i] / sarsa_steps_avg[i] for i in range(len(sarsa_rewards_avg))]
    
    plt.plot(q_learning_rewards_per_step_avg, label="Q-Learning", color='blue')
    plt.plot(sarsa_rewards_per_step_avg, label="SARSA", color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward Per Step')
    plt.title('Average Reward Per Step (Efficiency)')
    plt.legend()
    plt.show()

def plot_q_value_distribution(q_learning_q_table_avg, sarsa_q_table_avg, episode):
    all_q_values_q_learning = q_learning_q_table_avg.flatten()
    all_q_values_sarsa = sarsa_q_table_avg.flatten()
    
    plt.hist(all_q_values_q_learning, bins=50, alpha=0.5, label='Q-Learning', color='blue')
    plt.hist(all_q_values_sarsa, bins=50, alpha=0.5, label='SARSA', color='green')
    plt.xlabel('Q-Value')
    plt.ylabel('Frequency')
    plt.title(f'Q-Value Distribution at Episode {episode}')
    plt.legend()
    plt.show()

plt.plot(q_learning_rewards_avg, label="Q-Learning")
plt.plot(sarsa_rewards_avg, label="SARSA")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.show()

plot_steps_per_episode(q_learning_steps_avg, sarsa_steps_avg)

plot_q_table_convergence(q_learning_q_table_history_avg, sarsa_q_table_history_avg)

plot_reward_per_step(q_learning_rewards_avg, q_learning_steps_avg, sarsa_rewards_avg, sarsa_steps_avg)

plot_q_value_distribution(q_learning_q_table_history_avg[-1], sarsa_q_table_history_avg[-1], num_episodes)



# %%
