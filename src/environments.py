import gym
import numpy as np


class PongEnvironment:
    ENV_NAME = 'Pong-v4'

    # TODO: convert actions to ENUM
    ACTION_NOP = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTIONS = [ACTION_UP, ACTION_DOWN]

    def __init__(self):
        self._env = gym.make(self.ENV_NAME)

    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        episode_finished = False
        observations, actions, rewards = [], [], []
        observation = self._env.reset()
        while not episode_finished:
            if render:
                self._env.render()
            observations.append(observation)
            actions_probability = agent.predict(observation)
            actions.append(actions_probability)
            action = self._choose_probable_action(actions_probability)
            observation, reward, episode_finished, info = self._env.step(action)
            rewards.append(reward)
        return observations, actions, rewards

    def _choose_probable_action(self, probability):
        return np.random.choice(self.ACTIONS, p=probability)

    def _process_episode_rewards(self, rewards, gamma=0.99):
        """ Smooth reward for specific environment. """
        # TODO: gamma decay for Pong
        return rewards
