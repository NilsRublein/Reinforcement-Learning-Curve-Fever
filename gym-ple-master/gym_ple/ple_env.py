
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# changed ple_game=True to false
# Changed observation space from screen image to array of 12 vals
# Changed state from image to beams
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import os
import gym
from gym import spaces
from ple import PLE
import numpy as np

class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, ple_game=True, **kwargs):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        # open up a game state to communicate with emulator
        import importlib
        if ple_game:
            game_module_name = ('ple.games.%s' % game_name).lower()
        else:
            game_module_name = game_name.lower()
            #game_module_name = ('gym-curve-fever.gym_curve_fever.envs.%s' % game_name).lower()
            
            # C:\Users\Nils\Desktop\ML2\gym-curve-fever\gym_curve_fever\envs
            
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)(**kwargs)
        self.game_state = PLE(game, fps=30, display_screen=display_screen)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        
        print("action set: ",self._action_set)
        print("action space: ",self.action_space)
        
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.observation_space = spaces.Box(low=0, high=480, shape=(12,), dtype = np.uint8) # need right observation space! here, the 12 beam values 
        
        self.viewer = None

        self.obs_dim = self.observation_space.shape[0] # From Lilja

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        #state = self._get_image() 
        state = self.game_state.getGameState() # Returns player position (x & y), player angle, beams
        #state = np.reshape(state, [1, self.obs_dim])  # From Lilja
        
     
        
        # This bit transposes the 1D array of beams ...
        #state = np.array([state])
        #state = state.T
        
        terminal = self.game_state.game_over()
        test = "hallo"
        return state, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        self.observation_space = spaces.Box(low=0, high=480, shape=(12,), dtype = np.uint8) # Again, need only 12 values for the beams
        self.game_state.reset_game()
        #state = self._get_image()
        state = self.game_state.getGameState() # Beams
        
        
        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
