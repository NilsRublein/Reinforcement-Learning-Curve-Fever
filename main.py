
"""
This file runs a pygame implementation of Curve fever in a gym env using gym-ple.
The curve fever environment itself is located in PyGame-Learning-Environment-master\ple\games.

For RL algorithms we are using stable-baselines 3 (SB3):
    git:                https://github.com/DLR-RM/stable-baselines3 
    documentation:      https://stable-baselines3.readthedocs.io/en/master/

In which action space can I use my agent?
    Discrete            DQN
    Box                 DDPQ, SAC, TD3
    Discrete & Box      PPO, HER, A2C
    
TODO:
    - Graphs 
        - Fix x axis of plot_results()
        - Make a function to plot Reward vs loss (mean and std)
        - Confirm that different agents show different values in tensorboard. E.g., DQN doesn't show loss
    - What other models would be interesting and are inplemented in SB3?
    - Make seperate env with continuous action space
    - Hyperparameter tuning, see 
        https://github.com/optuna/optuna
        https://github.com/optuna/optuna/issues/1314 
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/callbacks.py#L10 -> use the file above in a callback?
        
        TL;DR
        See the section "Hyperparameter Tuning" on https://github.com/DLR-RM/rl-baselines3-zoo
        It says that hyperparameter tuning is not implemented for DQN, but if you look in hyperparams_opt.py, there is actually a function for it.
        Try this out, if it doesnt work, it shouldnt be too hard to implement it yourself based on the functions for the other algorithms like TD3
        
        
        
        

################################################################################################
############################################# MISC #############################################

Changed PLE, PLEEnv, envs themselves, registrations (PLE & gym-ple)
        
"""


#%% Load libraries etc.

import os, sys
import gym
import gym_ple

import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, A2C, TD3
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy


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
    Plot results (rewards)
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
    
def make_gif(name, frames=350, fps_ = 30):
    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    for i in range(frames):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode='rgb_array')
    
    imageio.mimsave(f'{name}.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=fps_)
    
def show_agent():
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

#%% 
if __name__ == '__main__':

    log_dir = "./dqn_logs/" # In this folder we will save the best model of our agent and all the logs for tensorbaord. Change the name for a different agent!
    #env_name = 'CurveFeverContinuous-v0' # For discrete action space, use 'CurveFeverDiscrete-v0'
    env_name = 'CurveFeverDiscrete-v0'
    env = gym.make(env_name if len(sys.argv)<2 else sys.argv[1])
    env = Monitor(env, log_dir)
    
    #env.seed(0) # uncomment for when you want actual random agent with untrained model

#%% Define callbacks
"""
SB3 callbacks:
https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
"""

eval_env = gym.make(env_name) # evaluation env
eval_callback = EvalCallback(eval_env, best_model_save_path= log_dir + 'best_model',
                             log_path= log_dir + 'results', eval_freq=500)

# Make a chain of callbacks 
callback = CallbackList([FigureRecorderCallback(), eval_callback])

#%% Define model and hyper parameters
hyper_params = {
    "buffer_size":              50000,
    "exploration_initial_eps":  1,
    "gamma":                    0.085,
    "exploration_fraction":     0.1,
    "learning_rate":           0.0001
    }

model = DQN('MlpPolicy', env, verbose=1, **hyper_params, tensorboard_log=log_dir)

# Note that A2C  takes different hyper_params then DQN, for now I just have it on default values for testing
# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir) 
#model = TD3('MlpPolicy', env, verbose=1,gamma=0.8, learning_rate=0.0001, tensorboard_log=log_dir) 

#%% eval untrained model
print("Starting to evaluate untrained (Random) model")
start = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
end = time.time()

print("Time elapsed in minutes: ", (end - start)/60)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#%% eval trained model
time_steps = 1000

print("Starting training")
start = time.time()
model.learn(total_timesteps=time_steps, log_interval=4, callback=callback)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
end = time.time()

print("Time elapsed in minutes: ", (end - start)/60)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#%% Plots
'''
How to plot with tensorboard:
After having trained the model, you can view stats in tensorboard. 
To do this, cd in your terminal to the folder of this file and enter
    tensorboard --logdir ./dqn_logs/
in your terminal. Make sure you are in the right python env!

For more info see: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
'''

plot_results(log_dir) # X axis is not correct ... don't know what is going wrong here yet.
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Curve Fever, Episode Rewards vs Timesteps") 
#results_plotter.plot_results([log_dir], 10000, results_plotter.X_EPISODES, "Curve Fever, Episode Rewards vs Episodes") 

#%% Show trained agent 
show_agent()
      
#%% Make a GIF of the trained agent  
make_gif('name', frames=200, fps_=60) 

#%% Load current best model (From this training run; actual best model is as of now DQN_gamma0_085 in ./models/)
path_2_best_model = log_dir + "best_model/best_model"
model = DQN.load(path_2_best_model)

#%% Save a model
name = "DQN_gamma0_085"
path = "./models/" + name
model.save(path)
