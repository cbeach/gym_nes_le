import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import nes_le
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
        assert obs_type in ('ram', 'image')

        self.game_path = nes_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.nes_le = nes_py.NESLEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        #assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        #self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        (screen_width, screen_height) = self.nes_le.getScreenDims()

        self._action_set = self.nes_le.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def _step(self, a):
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.nes_le.act(action)
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

ACTION_MEANING = {
    0 : "NOOP",
    1 : "UP",
    2 : "RIGHT",
    3 : "LEFT",
    4 : "DOWN",
    5 : "UPRIGHT",
    6 : "UPLEFT",
    7 : "DOWNRIGHT",
    8 : "DOWNLEFT",
    9 : "A",
    10 : "B",
    11 : "UPA",
    12 : "RIGHTA",
    13 : "LEFTA",
    14 : "DOWNA",
    15 : "UPRIGHTA",
    16 : "UPLEFTA",
    17 : "DOWNRIGHTA",
    18 : "DOWNLEFTA",
    19 : "UPB",
    20 : "RIGHTB",
    21 : "LEFTB",
    22 : "DOWNB",
    23 : "UPRIGHTB",
    24 : "UPLEFTB",
    25 : "DOWNRIGHTB",
    26 : "DOWNLEFTB",
    27 : "UPB",
    28 : "RIGHTB",
    29 : "LEFTB",
    31 : "DOWNAB",
    32 : "UPRIGHTAB",
    33 : "UPLEFTAB",
    34 : "DOWNRIGHTAB",
    35 : "DOWNLEFTAB",
}
