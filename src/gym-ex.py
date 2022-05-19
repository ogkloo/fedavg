import gym
import pygame

pygame.init()
screen = pygame.display.set_mode((1200,800))
# pygame.display.list_modes()

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)

env.close()
