
"""
This file runs a pygame implementation of Curve fever in a gym env using gym-ple.
The curve fever environment itself is located in PyGame-Learning-Environment-master\ple\games.

After having trained the model, you can view stats in tensorbaord. 
To do this, cd in your terminal to the folder of this file and enter 
tensorboard --logdir ./dqn_curve_fever/
in your terminal.



Questions for elena:
    1) Why does my model give me a mean reward of max 230, and if I train the model again with the same parameters, just max 100?
       Maybe this has something to do with exploration? Did I second not explore enough but in the first time?
    2) Why are there these dents in the graphs?
    3) Why does tensorboard only log single vals for this env, but does not for e.g. cartpole?
    
best values so far
    - 0.05
    - 0.085

TODO:
  
    - Clean up, write functions for e.g. monitoring, fix graphs & tensorboard
    - Exploration rate??
    
    - What other models would be interesting and are inplemented in SB3?
    - Hyperparameter tuning,check 
        https://github.com/optuna/optuna
        https://github.com/optuna/optuna/issues/1314 
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/callbacks.py#L10 -> use the file above in a callback?
        
        TL;DR
        See the section "Hyperparameter Tuning" on https://github.com/DLR-RM/rl-baselines3-zoo
        It says that hyperparameter tuning is not implemented for DQN, but if you look in hyperparams_opt.py, there is actually a function for it.
        Try this out, if it doesnt work, it shouldnt be too hard to implement it yourself based on the functions for the other algorithms like TD3
    - Make a dict for the params 
    - progress bar, https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/4_callbacks_hyperparameter_tuning.ipynb
    
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
#from gym.wrappers import Monitor # We wanna use the SB3 monitor
import gym_ple

import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
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
    x, y = ts2xy(load_results(log_folder), 'timesteps') # Can also do 'episodes instead'
    y = moving_average(y, window=50)
    # Truncate x    
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


if __name__ == '__main__':

    log_dir = "./dqn_curve_fever/" # Make sure you are in the rigt directory :)
    env = gym.make('CurveFever-v0' if len(sys.argv)<2 else sys.argv[1])
    env = Monitor(env, log_dir)
    
    env.seed(0) #uncomment for when you want actual random agent with untrained model
    #env = DummyVecEnv([lambda: env]) #This vectorizes the env, but that is already automatically done by SB3

#%% Define model and callbacks
"""
SB3 DQN:
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

This implementation provides only vanilla Deep Q-Learning and has no extensions such as 
Double-DQN, Dueling-DQN and Prioritized Experience Replay.

SB3 callbacks:
https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

"""

eval_env = gym.make('CurveFever-v0') # evaluation env
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                             log_path='./logs/results', eval_freq=500)

# Make a chain of callbacks
callback = CallbackList([FigureRecorderCallback(), eval_callback])
# callback = FigureRecorderCallback()

# TODO: Make a dict for the params 
# Best so far: gamma = 0.085
model = DQN('MlpPolicy', env, verbose=1, buffer_size=500000, exploration_initial_eps=1.0, gamma=0.99, exploration_fraction= 0.2, learning_rate=0.0001, tensorboard_log=log_dir)


#%% eval untrained model
print("Starting to evaluate untrained (Random) model")
start = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
end = time.time()
print("Time elapsed in minutes: ", (end - start)/60)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#%% eval trained model
print("Starting training")
start = time.time()
model.learn(total_timesteps=10000, log_interval=100, callback=callback)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
end = time.time()
print("Time elapsed in minutes: ", (end - start)/60)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#%% Load current best model (From this training run; actual best model is as of now DQN_gamma0_085 in ./models/)
path_2_best_model = "./logs/best_model/best_model"
model = DQN.load(path_2_best_model)

#%% more plots
plot_results(log_dir)
results_plotter.plot_results([log_dir], 10000, results_plotter.X_EPISODES, "Curve Fever, Episode Rewards vs Timesteps") #X_EPISODES , X_TIMESTEPS


#%% Show trained agent 
#env = Monitor(env, "recording", force=True)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
      
#%% Make a GIF
import imageio

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(500):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render(mode='rgb_array')

imageio.mimsave('Trained_gamma0_99.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

#%% Save a model
name = "DQN_gamma0_085"
path = "./models/" + name
model.save(path)
