from gym.envs.registration import register
register(
    #id='flpbird-v0',
    id = 'scienceCampBird-v1',
    entry_point='flappyBird.env:birdEnv',
)

from flappyBird.env import birdEnv
