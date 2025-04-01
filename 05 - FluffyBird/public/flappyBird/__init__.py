from gym.envs.registration import register
register(
    #id='flpbird-v0',
    id = 'scienceCampBird-v1',
    entry_point='public.flappyBird.env:birdEnv',
)

from public.flappyBird.env import birdEnv
