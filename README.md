# Reinforcement-Learning-Curve-Fever

This repo contains a reinforcement learning implementation of the game Curve fever and is based on the work of [A. Lilja and E. Bohnsack](https://github.com/erikbohnsack/reinforcement-achtung)
It makes use of `PLE` and `gym-ple` to create a custom `gym` env for our Curve Fever.

OpenAI `PLE` (PyGame Learning Environment) is a learning environment, mimicking the Arcade Learning Environment interface, allowing a quick start to Reinforcement Learning in Python. The goal of `PLE` is allow practitioners to focus design of models and experiments instead of environment design. Finally, `gym-ple` allows to use `PLE` as a gym environment.  

However, I had to modify some files of `PLE` and `gym-ple` to make the implementation work (e.g. fix bugs caused by pygamewrapper.py, alter registration files to include curve fever, etc.), therefore I included both modified repos in here. 

To install PLE and gym-ple, `cd` into their respective folders once you have downloaded them and then use 
```
pip install -e .
```
In addition, you will also need to add `ffmpeg` to your `path` variable for rendering, you can dowload for instance from [here](https://web.archive.org/web/20200916091820mp_/https://ffmpeg.zeranoe.com/builds/win64/shared/ffmpeg-4.3.1-win64-shared.zip).
If you don't know how to add it to your `path` variable, you can follow this [guide](https://windowsloop.com/install-ffmpeg-windows-10/).  

The curve fever environment itself is located in `PyGame-Learning-Environment-master\ple\games`.   
Convserion to a gym env is done via the file `gym-ple-master\gym_ple\ple_env.py`
The `main` file is where all the magic happens. Here you specify the env that you want to use (e.g. Curve Fever, Flappybird, etc.), and specify the agent that you want to train, etc.. Check out [link](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/1_getting_started.ipynb) for a tutorial on using stable baseline agents (e.g. DQN).

# TODO:

- implement other agents, see resources.
- implement Lilja's and Bohnsack's curve fever version using gym-ple
- take full frame as input via a CNN instead of beam search
- training against non-random opponents (e.g. version by Lilja & Bohnsack)

# Some resources

**Making a Custom gym env:**  
https://github.com/openai/gym/blob/master/docs/creating-environments.md

**PLE:**  
https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html  
https://github.com/ntasfi/PyGame-Learning-Environment

**ple-gym:**  
https://github.com/lusob/gym-ple

**Agents:**  
https://github.com/openai/gym/blob/master/docs/agents.md  
https://stable-baselines.readthedocs.io/en/master/
https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/1_getting_started.ipynb
