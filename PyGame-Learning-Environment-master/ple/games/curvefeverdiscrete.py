# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:47:42 2020

@author: Nils
"""

# -*- coding: utf-8 -*-
"""
See PLE documentation:
    https://pygame-learning-environment.readthedocs.io/en/latest/user/tutorial/adding_games.html\
        
- PLE (PyGame Learning Environment) is a learning environment, mimicking the Arcade Learning Environment interface, allowing a quick start to Reinforcement Learning in Python
- Then, use gym-ple to use PLE as a gym environment.
- Follow the instructions here, to install PLE and gym-ple: https://github.com/lusob/gym-ple
    
"""

#%% Init libraries & variables
import pygame
import sys
import math
import random
import numpy as np
from ple.games import base
#from ple import PLE

import gym
from gym import spaces
from pygame.constants import KEYDOWN, KEYUP, K_F15
from pygame.constants import K_w, K_a, K_s, K_d

#Far by
WINWIDTH = 480      # width of the program's window, in pixels
WINHEIGHT = 480     # height in pixels
TEXT_SPACING = 130
RADIUS = 2          # radius of the circles
PLAYERS = 1         # number of players
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
P1COLOUR = RED
P2COLOUR = GREEN
P3COLOUR = BLUE

# Adam
SKIP_PROBABILITY = 0.01
SKIP_COUNT = 4
SPEED_CONSTANT = 2

BG_COLOR = (25, 25, 25)
BEAM_SIGHT = 240 # length of the beams
BEAM_MAX_ANGLE = 120
BEAM_STEP = 30
BEAMS = range(BEAM_MAX_ANGLE, -BEAM_MAX_ANGLE-BEAM_STEP, -BEAM_STEP)


#%% Player class
class Player:
    def __init__(self, color, width):
        self.color = color
        self.score = 0
        self.skip = 0
        self.skip_counter = 0
        # generates random position and direction
        self.width = width
        self.x = random.randrange(50, WINWIDTH - WINWIDTH/4)
        self.y = random.randrange(50, WINHEIGHT - WINHEIGHT/4)
        self.angle = random.randrange(0, 360)
        self.sight = BEAM_SIGHT
        self.beams = np.ones(len(BEAMS)) * BEAM_SIGHT
        
    def update(self,dt):
        # computes current movement
        if self.angle > 360:
            self.angle -= 360
        elif self.angle < 0:
            self.angle += 360
        self.x += int(RADIUS * SPEED_CONSTANT * math.cos(math.radians(self.angle)))
        self.y += int(RADIUS * SPEED_CONSTANT * math.sin(math.radians(self.angle)))

    def beambounce(self, current_angle, screen):
        # Checks if beams bounce against sth and return distance
        # As far as i can see, this only checks for the walls not for the player or other players, amirite?
        # Maybe the bit where they check for BG, if there is a different colour, it must be the player itself or a different player
        # Could optimize this if we want to include strategies to let other players loose rather than just trying to survive
        _distance = self.sight
        for i in range(1, self.sight + 1):
            _x = self.x + i * int(RADIUS * SPEED_CONSTANT * math.cos(math.radians(current_angle)))
            _y = self.y + i * int(RADIUS * SPEED_CONSTANT * math.sin(math.radians(current_angle)))
            
            # Check for walls 
            x_check = (_x <= 0) or (_x >= WINWIDTH)
            y_check = (_y <= 0) or (_y >= WINHEIGHT)

            if (x_check or y_check):
                d_bounce = True
            else:
                d_bounce = screen.get_at((_x, _y)) != BG_COLOR

            if d_bounce or i == self.sight:
                _distance = int(np.round(math.sqrt((self.x - _x) ** 2 + (self.y - _y) ** 2)))
                break

        return _distance
    
    def beam(self, screen):
        # Return the length of each beam and make sure the angle stays in range
        for index, angle in enumerate(BEAMS):
            current_angle = self.angle + angle
            if current_angle > 360:
                current_angle -= 360
            elif current_angle < 0:
                current_angle += 360
            self.beams[index] = self.beambounce(current_angle, screen)

    def draw(self, screen):
        if self.skip:
            self.skip_counter += 1
            if self.skip_counter == SKIP_COUNT:
                self.skip_counter = 0
                self.skip = 0
        elif random.random() < SKIP_PROBABILITY:
            self.skip = 1
        else:
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.width)
            
            

        

#%% Env Class
# Check if it is base.Game or base.Pywrapper
class CurveFeverDiscrete(base.PyGameWrapper):

        # pass the width, height and valid actions the game responds too.
        # Recommended to have height = width, don't know why tho.
        def __init__(self, width=480, height=480):

                actions = {
                        "left": K_a,
                        "right": K_d
                }
                
                # The game must inherit from base.Game as it sets attributes and methods 
                # used by PLE to control game flow, scoring and other functions.
                base.PyGameWrapper.__init__(self, width, height, actions=actions, discrete = True)

                self.discrete = True
                self.last_action = []
                self.action = []
                self.observation_space = spaces.Box(low=0, high=WINWIDTH, shape=(12,), dtype = np.uint8)
                
                self.rewards = {    # TODO: take as input
                            "positive": 1.0,
                            "negative": -1.0,
                            "tick": 1,
                            "loss": 0,
                            "win": 5.0
                        }
              
                # Other
                self.rng = None #random number generator (?)
                self.score = 0.0  # required.
                self.lives = 0    # required. Can be 0 or -1 if not required.
                self.ticks = 0
                self.previous_score = 0
              
                # Screen & FPS stuff
                self.height = height
                self.width = width
                self.screen_dim = (width, height)  # width and height
                self.BG_COLOR = BG_COLOR
                #pygame.init()
                #self.my_font = pygame.font.SysFont('arial', 37) #bauhaus93
                
                # Not sure if needed
                self.viewer = None
                self.allowed_fps = None  # fps that the game is allowed to run at.
                
                self._setup()
                self.init()
                
        ##########################################################################
        # First we cover all four required methods: init, getScore, game_over, and step. 
        # These methods are all required to interact with our game.         
                
        def init(self):
                """
                Sets the game to a clean state. The minimum this method must do is to reset the self.score attribute of the game.
                It is also strongly recommended this method perform other game specific functions such as player position and clearing the screen.
                This is important as the game might still be in a terminal state if the player and object positions are not reset; 
                which would result in endless resetting of the environment.
                """
                
                self.player = Player(GREEN, RADIUS)
                self.screen.fill(self.BG_COLOR)
                self.score = 0
                self.ticks = 0
                self.lives = 1
                self.sight = BEAM_SIGHT

        def getScore(self):
                # Returns the current score of the agent.
                return self.score

        def game_over(self):
                # Must return True if the game has hit a terminal state.
                return self.lives == -1 

        def step(self, dt):
            
                """
                Responsible for the main logic of the game. It is called everytime our agent performs an action on the game environment.
                step() performs a step in game time equal to dt (delta time), the amount of time elapsed since the last frame in milliseconds. 
                dt is required to allow the game to run at different frame rates, such that the movement speeds of objects are scaled by elapsed time. 
                
                With that said the game can be locked to a specific frame rate, by setting self.allowed_fps, and written such that 
                step moves game objects at rates suitable for the locked frame rate. The function signature always expects dt to be passed, 
                the game logic does not have to use it though.
                """
                
                dt /= 1000.0 # Not actually using this for now ... 
                
                self.ticks += 1
                #self.screen.fill(self.BG_COLOR)
                self._handle_player_events()
                self.score += self.rewards["tick"]
                
                self.player.update(dt)
                if self.collision(self.player.x, self.player.y, self.player.skip):
                    self.lives = -1
                
                self.player.beam(self.screen)
                self.player.draw(self.screen)

                # print("x and y position: ", self.player.x, self.player.y)
            
        ########################################################################## 
        # More game specific stuff
        
        def _handle_player_events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
    
                if event.type == pygame.KEYDOWN:
                    key = event.key
                    if key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
    
                    if key == self.actions["left"]:
                        self.player.angle -= 10
    
                    if key == self.actions["right"]:
                        self.player.angle += 10
            
        def getGameState(self):
            state = np.hstack(([self.player.x, self.player.y, self.player.angle], self.player.beams))
            #print("getGameState: ", state)
            return state
        
        def collision(self, x, y, skip):
            collide_check = 0
            try:
                x_check = (x < 0) or \
                          (x > self.width)
                y_check = (y < 0) or \
                          (y > self.height)
    
                collide_check = self.screen.get_at((x, y)) != self.BG_COLOR
            except IndexError:
                x_check = (x < 0) or (x > self.width)
                y_check = (y < 0) or (y > self.height)
    
            if skip:
                collide_check = 0
            if x_check or y_check or collide_check:
                return True
            else:
                return False
       
        def reset(self):
            self.observation_space = spaces.Box(low=0, high=WINWIDTH, shape=(12,), dtype=np.uint8)
            self.last_action = []
            self.action = []
            self.previous_score = 0.0
            self.init()
            state = self.getGameState()
            return state
        
        def seed(self, seed):
            rng = np.random.RandomState(seed)
            self.rng = rng
            self.init()
        
        #################################################################
    
        
#%% Main
if __name__ == "__main__":
    import numpy as np

    pygame.init()
    game = CurveFeverDiscrete(width=WINWIDTH, height=WINHEIGHT)
    game.rng = np.random.RandomState(24)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()

        game.step(dt)
        pygame.display.update()