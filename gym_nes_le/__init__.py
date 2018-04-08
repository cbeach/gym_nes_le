from gym.envs.registration import register

for game in ['super_mario_brothers',]
    # No frameskip. (NES has no entropy source, so these are
    # deterministic environments.)
    register(
        id='{}NoFrameskip'.format(name),
        entry_point='gym.envs.nes:NESEnv',
        kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
        max_episode_steps=frameskip * 100000,
        nondeterministic=nondeterministic,
    )



