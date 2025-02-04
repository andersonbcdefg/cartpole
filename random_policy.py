import gymnasium as gym
import time

env = gym.make('CartPole-v1', render_mode='human')
observation, info = env.reset()

episode = 0
total_steps = 0

for step in range(1000):
    time.sleep(0.01)  # Slow it down so we can watch
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    if terminated or truncated:
        print(f"Episode {episode} finished after {total_steps} steps. Reason: {}")
        episode += 1
        total_steps = 0
        observation, info = env.reset()

env.close()
