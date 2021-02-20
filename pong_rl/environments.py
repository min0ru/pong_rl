import abc
import gym
import numpy as np

from .actions import PongAction


class BasePongEnvironment(abc.ABC):
    """ Base Class for Pong Environments. """
    ENV_NAME = 'Pong-v4'
    OBSERVATION_SHAPE = (80, 80)

    def __init__(self):
        self.action_values = [action.value for action in PongAction]

    def _process_observations(self, observations):
        """ Extract area of interest 160x160 with down sampling and converting to b/w colors. """
        images = observations[:, 34:194, :, :]  # Cut out the area of interest
        images = images[:, ::2, ::2, 1]  # Down sample to 80x80 and single channel
        images[images < 100] = 0
        images[images > 0] = 255
        return images

    def _choose_probable_actions(self, probabilities):
        """ Choose available action """
        return [np.random.choice(self.action_values, p=p) for p in probabilities]

    @staticmethod
    def _process_episode_rewards(rewards, gamma=0.99):
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
        """ Play episode using agent and return observations, actions, rewards, total score. """
        pass


class PongEnvironment(BasePongEnvironment):
    """ Pong Environment Class. """

    def __init__(self):
        """ Initialize Pong gym environment. """
        super().__init__()
        self._env = gym.make(self.ENV_NAME)

    @staticmethod
    def _expand_observations(observations):
        """ Convert observation to the array of observations if needed. """
        if len(observations.shape) < 4:
            observations = observations[np.newaxis]
        return observations

    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        episode_finished = False
        output_observations, actions, rewards = [], [], []
        observation = self._env.reset()
        while not episode_finished:
            if render:
                self._env.render()
            observations = self._expand_observations(observation)
            observations = self._process_observations(observations)
            output_observations.append(observations[0])
            action_probabilities = agent.predict(observations)
            actions.append(action_probabilities[0])
            action = self._choose_probable_actions(action_probabilities)
            observation, reward, episode_finished, info = self._env.step(action)
            rewards.append(reward)
        output_observations = np.array(output_observations)
        actions = np.array(actions)
        processed_rewards = self._process_episode_rewards(rewards)
        score = sum(rewards)
        return output_observations, actions, processed_rewards, score


class VectorizedPongEnvironment(BasePongEnvironment):
    """ Pong Environment Class. """

    def __init__(self, num_environments=4):
        """ Initialize Pong gym environment. """
        super().__init__()
        self.parallel_environments = num_environments
        self._env = gym.vector.make(self.ENV_NAME, num_environments)

    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        episode_finished = False
        environments_finished_episode = 0
        output_observations, actions, rewards = [], [], []
        observations = self._env.reset()
        while environments_finished_episode < self.parallel_environments:
            observations = self._process_observations(observations)
            output_observations.extend(observations)
            action_probabilities = agent.predict(observations)
            actions.extend(action_probabilities)
            action = self._choose_probable_actions(action_probabilities)
            observations, reward, episodes_finished, infos = self._env.step(action)
            environments_finished_episode += np.sum(episodes_finished)
            rewards.extend(reward)
        output_observations = np.array(output_observations)
        actions = np.array(actions)
        processed_rewards = self._process_episode_rewards(rewards)
        score = sum(rewards)
        return output_observations, actions, processed_rewards, score
