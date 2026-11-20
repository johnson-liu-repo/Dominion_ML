'''
Simple starting environment.
python3.14.7
gymnasium-1.4.0
'''



import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DominionBuyEnv(gym.Env):
    def __init__( self,
                  player ):

        super().__init__()

        # TODO: Define action_space

        # Simple action space with 8 possible actions.
        # 0: Pass
        # 1: Buy Copper
        # 2: Buy Silver
        # 3: Buy Gold
        # 4: Buy Estate
        # 5: Buy Duchy
        # 6: Buy Province
        # 7: Buy Curse

        self.action_space = spaces.Discrete(8)

        # TODO: Define observation_space

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (3,),
            dtype = np.float32
        )

        self.player = player


    def _get_obs(self):
        money = self.player.state.money
        victory_points = self.player.get_victory_points()
        turns = self.player.turns

        observation = np.array(
            [ money, victory_points, turns ],
              dtype = np.float32
        )

        return observation



    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)

        # TODO: Initialize a new Dominion game

        # TODO: Return observation and info

        raise NotImplementedError

    def step(self, action):

        # TODO: Translate action into a purchase

        # TODO: Advance the game

        # TODO: Return Gymnasium's five outputs

        raise NotImplementedError


if __name__ == "__main__":

    from pyminion_master.pyminion.human import Human
    from pyminion_master.pyminion.game import Game

    player = Human(player_id="rl_agent")

    game = Game(
        players = [player],
        expansions = [],
        random_order = False
    )

    game.start()

    env = DominionBuyEnv(player)

    observation = env._get_obs()

    print(observation)
    print(observation.shape)
    print(env.observation_space.contains(observation))