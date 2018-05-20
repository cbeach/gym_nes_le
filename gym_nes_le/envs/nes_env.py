import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import nes_le
    from nes_le.interface import NESLEInterface, show_image
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install NES dependencies by running 'pip install gym[nes]'.)".format(e))

import logging
logger = logging.getLogger(__name__)

class NESEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='super_mario_bros', obs_type='image', frameskip=(2, 5), repeat_action_probability=0.):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type)
        assert obs_type in ('image')

        self._obs_type = obs_type
        self.frameskip = frameskip
        self.nes_le = NESLEInterface(game)
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        #assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        #self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        (screen_width, screen_height) = self.nes_le.getScreenDims()

        self._action_set = self.nes_le.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))

    def _step(self, a):
        reward = 0.0

        # TODO: re-implement seeding and randomized actions
        # if isinstance(self.frameskip, int):
        #     num_steps = self.frameskip
        # else:
        #     num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        num_steps = 1
        for _ in range(num_steps):
            reward += self.nes_le.act(a)
        ob = self._get_obs()

        return ob, reward, self.nes_le.game_over(), {"nes_le.lives": self.nes_le.lives()}

    def _get_image(self):
        return self.nes_le.getScreenRGB()

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        return self._get_image()

    # return: (states, observations)
    def _reset(self):
        self.nes_le.reset_game()
        return self._get_obs()

    def _render(self, mode='human', close=False):
        frame = self._get_image()
        if mode == 'rgb_array':
            return frame
        elif mode == 'human':
            show_image(frame)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    # TODO: implement these from front to back
    #def clone_state(self):
    #    """Clone emulator state w/o system state. Restoring this state will
    #    *not* give an identical environment. For complete cloning and restoring
    #    of the full state, see `{clone,restore}_full_state()`."""
    #    state_ref = self.ale.cloneState()
    #    state = self.ale.encodeState(state_ref)
    #    self.ale.deleteState(state_ref)
    #    return state

    #def restore_state(self, state):
    #    """Restore emulator state w/o system state."""
    #    state_ref = self.ale.decodeState(state)
    #    self.ale.restoreState(state_ref)
    #    self.ale.deleteState(state_ref)

    #def clone_full_state(self):
    #    """Clone emulator state w/ system state including pseudorandomness.
    #    Restoring this state will give an identical environment."""
    #    state_ref = self.ale.cloneSystemState()
    #    state = self.ale.encodeState(state_ref)
    #    self.ale.deleteState(state_ref)
    #    return state

    #def restore_full_state(self, state):
    #    """Restore emulator state w/ system state including pseudorandomness."""
    #    state_ref = self.ale.decodeState(state)
    #    self.ale.restoreSystemState(state_ref)
    #    self.ale.deleteState(state_ref)

print(dir(nes_le.interface))
ACTION_MEANING = {key: value[0] for key, value in nes_le.interface.NESLEInterface.actions.items()}

if __name__ == "__main__":
    n = NESEnv()
    while True:
        n._step(0)
        show_image(n._render(mode='rgb_array'))
