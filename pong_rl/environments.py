import abc

import gym
import numpy as np

from .actions import PongAction
from .storages import EpisodeStorage


class BasePongEnvironment(abc.ABC):
    """ Base Class for Pong Environments. """

    ENV_NAME = "Pong-v4"
    OBSERVATION_SHAPE = (80, 80)

    def __init__(self):
        self.action_values = np.array([action.value for action in PongAction])

    def _process_observations(self, observations):
        """ Extract area of interest 160x160 with down sampling and converting to b/w colors. """
        images = observations[:, 34:194, :, :]  # Cut out the area of interest
        images = images[:, ::2, ::2, 1]  # Down sample to 80x80 and single channel
        images[images < 100] = 0
        images[images > 0] = 255
        return images

    def _choose_probable_actions(self, probabilities):
        """Sample (random choice) environment actions using given predicted probabilities.

        Return selected action and onehot vector with action index.
        """
        onehots = np.eye(len(self.action_values))
        indexes = np.arange(len(self.action_values))
        samples = [np.random.choice(indexes, p=p) for p in probabilities]
        actions = self.action_values[samples]
        action_onehots = onehots[samples]
        return actions, action_onehots

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
    """ Pong Environment Class. Single Threaded. """

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
            # actions.append(action_probabilities[0])
            action, onehot = self._choose_probable_actions(action_probabilities)
            actions.append(onehot)

            observation, reward, episode_finished, info = self._env.step(action)

            rewards.append(reward)
        output_observations = np.array(output_observations)
        actions = np.array(actions)
        processed_rewards = self._process_episode_rewards(rewards)
        rewards = np.array(rewards)
        score = np.sum(rewards[rewards >= 1.0])
        return output_observations, actions, processed_rewards, score


class VectorizedPongEnvironment(BasePongEnvironment):
    """ Pong Environment Class. Vectorized (parallel environments). """

    def __init__(self, num_environments=4):
        """ Initialize Pong gym environment. """
        super().__init__()
        self.num_environments = num_environments
        self._env = gym.vector.make(self.ENV_NAME, num_environments)

    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        results = EpisodeStorage(self.num_environments)
        score = 0
        environments_finished_episodes = 0

        observations = self._env.reset()
        while environments_finished_episodes < self.num_environments:
            if render:
                self._env.render()

            processed_observations = self._process_observations(observations)

            # Predict actions using agent
            action_probabilities = agent.predict(processed_observations)
            action, action_onehot = self._choose_probable_actions(action_probabilities)

            # Make a step and save results to vectorized storage
            observations, rewards, episodes_finished, infos = self._env.step(action)
            results.add(processed_observations, action_onehot, rewards, infos)
            score += rewards
            environments_finished_episodes += np.sum(episodes_finished)

        observations, actions, rewards, infos = results.get()
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = self._process_episode_rewards(rewards)
        return observations, actions, rewards, score
