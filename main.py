
"""
This file runs a pygame implementation of Curve fever in a gym env using gym-ple.
The curve fever environment itself is located in PyGame-Learning-Environment-master\ple\games.



!!!!!!!!!!!!!!!

set exploration higher, it should do more random shit ?

compare with untrained ... There is a difference, yasss


0.05 is really good, does sort of naruto sign

!!!!!!!!!!!!!!










0.45 was quite alright, 0.2 too
0.65-0.95 just did straight lines

0.01 does a very funky brezel kind of, it never changes direction tho, only one direction or noop

0.001 mostly straightlines, sometimes does a turn when approaching edge

Try out somewhere between 0.01 and 0.2?

0.05 curve actually goes up :D 2nd its good from the start then has a large dent


TODO:
  
    - What other models would be interesting and are inplemented in SB3?
    - Hyperparameter tuning
    - Visualization of results
    - Make a dict for the params 
    
    - Look into vectorization of env, normalization of action space, multiprocessing
    - can train in one line
    
Misc
    - Has a series on implementing DQN and other algorithms
    - https://medium.com/analytics-vidhya/reinforcement-learning-d3qn-agent-with-prioritized-experience-replay-memory-6d79653e8561
    
"""


#%%

import logging
import os, sys
import gym
from gym.wrappers import Monitor
import gym_ple

import time
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

#rom agentStuff import DQNAgent
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
    
class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    log_dir = "./dqn_curve_fever/" # Make sure you are in the rigt directory :)
    env = gym.make('CurveFever-v0' if len(sys.argv)<2 else sys.argv[1])
    env = Monitor(env, log_dir)

    #env = gym.make('CurveFeverFullImage-v0' if len(sys.argv)<2 else sys.argv[1])
    # env = DummyVecEnv([lambda: env])
    
#%% Implementation from Bohnsack & Lilja

    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(state_size=obs_dim, action_size=num_actions)
     
    outdir = '/tmp/random-agent-results'
    #env = Monitor(env, directory=outdir, force=True)
    env = Monitor(env, "recording", force=True)
    # https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/

    env.seed(0)
    #agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False
    fitnessValues = np.zeros(episode_count)

    for i in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, obs_dim])
        fitness = 0
        
        while True:
            env.render()

            # Decide action
            action = agent.act(state)

            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, obs_dim])
            fitness += reward

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state

            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, reward: {}".format(i, episode_count, fitness))
                fitnessValues[i] = fitness
                break
            # train the agent with the experience of the episode
        agent.replay(8)

    plt.plot(fitnessValues)
    plt.ylabel('Fitness')
    plt.xlabel('Epoch')
    plt.show()

    # Dump result info to disk
    env.close()
    print("Finished training.")

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)
    # Syntax for uploading has changed
    
#%% SB3 DQN
"""
Based on:
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

Note:
This implementation provides only vanilla Deep Q-Learning and has no extensions such as 
Double-DQN, Dueling-DQN and Prioritized Experience Replay.
"""

# TODO: Make a dict for the params 
model = DQN('MlpPolicy', env, verbose=1, buffer_size=500000, exploration_initial_eps=1.0, gamma=0.05, exploration_fraction= 0.2, learning_rate=0.0001, tensorboard_log=log_dir )

#%% eval untrained model
print("Starting untrained (Random) model")
start = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
end = time.time()
print("Time elapsed in minutes: ", (end - start)/60)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#%% eval trained model
print("Starting training")
start = time.time()
model.learn(total_timesteps=10000, log_interval=100, callback=FigureRecorderCallback())
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
end = time.time()
print("Time elapsed in minutes: ", (end - start)/60)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

"""
Saving and loading a model
model.save("dqn_pendulum")
del model # remove to demonstrate saving and loading
model = DQN.load("dqn_pendulum")

untrained   109.45 +/- 71.62

trained
@10000,     113.16 +/- 76.12


THing is, std is super high. New val should be multiple times the one of the std ...

waiiiit why is the initial exploration rate 0.05? Shouldnt it be decreasing from 1 to 0.05??
However, it doesnt make perfect circles anymore!
Still, it seems like it keeps repeating on action too many times, it either goes straight for too long or makes straight a circle ..


gamma (float) – the discount factor 
"discount factor essentially determines how much the reinforcement learning agents cares about rewards in the distant future relative to those in the immediate future. 
If γ=0, the agent will be completely myopic and only learn about actions that produce an immediate reward."
-> This was 0.99 by default ....


learning_rate
buffer_size (int) – size of the replay buffer
-> sort of saves history / experience. This might be too small / large? We are not doing anything with collect_rollouts(), which Collect experiences and store them into a ReplayBuffer.
Initially we started with 5000

... but are we using the experiences actually.?? 

with 50000 we have
untrained mean_reward:39.14 +/- 7.45
trained mean_reward:74.94 +/- 31.64

learning_starts (int) – how many steps of the model to collect transitions for before learning starts

"""
#%% more plots

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    print("1x: ",len(x))
    x = x[len(x) - len(y):]
    print("2x: ", len(x))

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir)
results_plotter.plot_results([log_dir], 10000, results_plotter.X_TIMESTEPS, "Curve Fever, Episode Rewards vs Timesteps")


#%% Show trained agent 
#env = Monitor(env, "recording", force=True)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
      
