import abc
import gym
import numpy as np

from .actions import PongAction


class BasePongEnvironment(abc.ABC):
    """ Base Class for Pong Environments. """
    ENV_NAME = 'Pong-v4'

    def __init__(self):
        self.action_values = [action.value for action in PongAction]

    def _process_observation(self, observation):
        """ Extract area of interest 160x160 with down sampling and converting to b/w colors. """
        image = observation[34:194, :, :]  # Cut out the area of interest
        image = image[::2, ::2, 1]  # Down sample to 80x80 and single channel
        image[image < 100] = 0
        image[image > 0] = 255
        return image

    def _choose_probable_action(self, probability):
        """ Choose available action """
        return np.random.choice(self.action_values, p=probability)

    def _process_episode_rewards(self, rewards, gamma=0.99):
        """ Smooth reward for specific environment. """
        processed_rewards = np.zeros_like(rewards, dtype=np.float32)
        sliding_sum = 0
        for i, reward in reversed(list(enumerate(rewards))):
            if reward != 0:
                sliding_sum = 0
            sliding_sum = sliding_sum * gamma + reward
            processed_rewards[i] = sliding_sum
        return processed_rewards

    @abc.abstractmethod
    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        pass


class PongEnvironment(BasePongEnvironment):
    """ Pong Environment Class. """

    def __init__(self):
        """ Initialize Pong gym environment. """
        super().__init__()
        self._env = gym.make(self.ENV_NAME)

    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        episode_finished = False
        observations, actions, rewards = [], [], []
        observation = self._env.reset()
        while not episode_finished:
            if render:
                self._env.render()
            observation = self._process_observation(observation)
            observations.append(observation)
            actions_probability = agent.predict(observation)
            actions.append(actions_probability)
            action = self._choose_probable_action(actions_probability)
            observation, reward, episode_finished, info = self._env.step(action)
            rewards.append(reward)
        observations = np.array(observations)
        actions = np.array(actions)
        processed_rewards = self._process_episode_rewards(rewards)
        score = sum(rewards)
        return observations, actions, processed_rewards, score
