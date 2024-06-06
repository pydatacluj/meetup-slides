import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 3000

SHOW_EVERY = 200

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def smooth(scalars, weight: float):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def get_discrete_state(continuous_state):
    discrete_s = (continuous_state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_s.astype(np.int32))


rewards = []
for episode in range(EPISODES):
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    frames = []
    done = False
    episode_reward = 0
    nr_steps = 0
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, truncated, terminated, _ = env.step(action)
        episode_reward += reward
        done = truncated or terminated
        new_discrete_state = get_discrete_state(new_state)
        nr_steps+=1
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Goal reached in {episode} in {nr_steps} steps")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    rewards.append(episode_reward)

env.close()
plt.plot(smooth(rewards, 0.99))
plt.show()

# now let's see how it performs
env = gym.make("MountainCar-v0", render_mode='human')
discrete_state = get_discrete_state(env.reset()[0])

done = False
while not done:
    discrete_state = get_discrete_state(env.state)
    action = np.argmax(q_table[discrete_state])
    new_state, _, done, _, _ = env.step(action)

