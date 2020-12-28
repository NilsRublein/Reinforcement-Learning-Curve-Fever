
"""
This file runs a pygame implementation of Curve fever in a gym env using gym-ple.
The curve fever environment itself is located in PyGame-Learning-Environment-master\ple\games.

"""

import logging
import os, sys
import gym
from gym.wrappers import Monitor
import gym_ple

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of output.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('CurveFever-v0' if len(sys.argv)<2 else sys.argv[1])
    # env = DummyVecEnv([lambda: env])
    
#%%

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
  
    outdir = '/tmp/random-agent-results'
    #env = Monitor(env, directory=outdir, force=True)
    env = Monitor(env, "recording", force=True)
    # https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        print("Current episode: ", i)
        while True:
            env.render() # env.render('rgb_array')
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    env.close()
    print("Finished training.")

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir)
    # Syntax for uploading has changed
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



