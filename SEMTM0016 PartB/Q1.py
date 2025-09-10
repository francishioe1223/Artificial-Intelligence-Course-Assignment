from envs.simple_dungeonworld_env import DungeonMazeEnv
import random
import numpy as np

SIZE = 8
env = DungeonMazeEnv(render_mode="human", grid_size=SIZE)

def random_policy(observation):
    return random.choice([0, 1, 2]) 

def all_forward_policy(observation):
    return 2  # 2=move forwards

def custom_policy(observation):
    robot_pos = tuple(observation["robot_position"])  
    camera_view = observation["robot_camera_view"]

    if np.all(camera_view > 0):     
            return 2  
    return random.choice([0, 1])

def rollout(env, policy, max_steps=100):

    observation, _ = env.reset()
    trajectory = []  
    cumulative_reward = 0

    for _ in range(max_steps):
       
        action = policy(observation)  
        observation, reward, terminated, _, _ = env.step(action)
        cumulative_reward += reward 
        trajectory.append((observation, action, reward))
       
        if terminated:
            break
    return trajectory, cumulative_reward


print("Start Random Policy")
trajectory, cumulative_reward = rollout(env, random_policy)

for state, action, reward in trajectory:
    print(f"State: {state}, Action: {action}, Reward: {reward}")

print(f"Cumulative reward: {cumulative_reward}")
print("End Random Policy")

print("Start All Forward Policy")
trajectory, cumulative_reward = rollout(env, all_forward_policy)

for state, action, reward in trajectory:
    print(f"State: {state}, Action: {action}, Reward: {reward}")

print(f"Cumulative reward: {cumulative_reward}")
print("End All Forward Policy")

print("Start Custom Policy")
trajectory, cumulative_reward = rollout(env, custom_policy)

for state, action, reward in trajectory:
    print(f"State: {state}, Action: {action}, Reward: {reward}")

print(f"Cumulative reward: {cumulative_reward}")
print("End Custom Policy")