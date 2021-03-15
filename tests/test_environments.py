import abc
import unittest

import numpy as np
from scipy.special import softmax

from pong_rl.agents import PongAgentRandom
from pong_rl.environments import PongEnvironment, VectorizedPongEnvironment


class MockPongAgent(PongAgentRandom):
    """ Mock Pong Agent that implements random acton choice. """

    @property
    def summary(self):
        """ Agent description. """
        return "RandomPongAgent as Mock Agent"


class AbstractPongEnvironmentCase(abc.ABC):
    ENV = None
    AGENT = None

    def setUp(self) -> None:
        """ Initialize Environment. """
        self.env = self.ENV()
        self.agent = self.AGENT(self.env.actions_len, self.env.observation_shape)

    def test_actions_len(self):
        """ Environment should return action space length. """
        actions_len = self.env.actions_len
        self.assertGreater(actions_len, 0, "Environment should return actions space length.")

    def test_observation_shape(self):
        """ Environment should return processed observations shape. """
        observation_shape = self.env.observation_shape
        self.assertGreater(
            len(observation_shape), 0, "Environment should return observations shape."
        )

    def test_process_episode_rewards(self):
        """ Test episode rewards smoothing method for Pong. """
        gamma = 0.99
        testing_rewards = [
            ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
            ([1, 0, 0, 0, 0], [1.0, 0, 0, 0, 0]),
            ([0, 0, 0, 0, -1], [-0.96059601, -0.970299, -0.9801, -0.99, -1.0]),
            ([0, 0, 0, 0, 1], [0.96059601, 0.970299, 0.9801, 0.99, 1.0]),
            (
                [0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
                [
                    0.96059601,
                    0.970299,
                    0.9801,
                    0.99,
                    1.0,
                    -0.96059601,
                    -0.970299,
                    -0.9801,
                    -0.99,
                    -1.0,
                ],
            ),
        ]

        for rewards, target_rewards in testing_rewards:
            rewards, target_rewards = np.array(rewards), np.array(target_rewards)
            processed_rewards = self.env._process_episode_rewards(rewards, gamma)
            np.testing.assert_array_almost_equal(processed_rewards, target_rewards)

    def test_environment_output(self):
        """ Play one episode with RandomAgent and test that env output is not empty. """
        observations, actions, rewards, score = self.env.play_episode(self.agent)
        observations_num = len(observations)
        actions_num = len(actions)
        rewards_num = len(rewards)

        # Pong score sum should be equal to a sum of rewards equal to 1.0
        positive_rewards = np.sum(rewards[rewards == 1.0])
        self.assertEqual(np.sum(score), positive_rewards)

        self.assertEqual(observations_num, actions_num)
        self.assertEqual(observations_num, rewards_num)
        self.assertGreater(observations_num, 0)

    def test_environment_rewards(self):
        """ Play episode and test environment rewards output. Rewards should be ordered. """
        _, _, rewards, _ = self.env.play_episode(self.agent)

        for p, n in zip(rewards, rewards[1:]):
            if (p * n) > 0 and abs(p) != 1.0:
                self.assertGreater(abs(n), abs(p))

    def test_choose_probable_actions(self):
        """Test environment action sampling method.

        Action sampler should return action from environment action space and onehot vector
        with selected action positional index.
        """
        num_probabilities = 10
        num_actions = len(self.env.action_values)
        probabilities = softmax(np.random.randn(num_probabilities, num_actions), axis=1)
        actions, onehots = self.env._choose_probable_actions(probabilities)

        # Check that for all input values there is corresponding actions and onehots.
        self.assertEqual(len(actions), len(probabilities))
        self.assertEqual(len(onehots), len(probabilities))

        # Check all selected actions are valid values from env action space.
        for action in actions:
            self.assertTrue(action in self.env.action_values)

        # Check that all onehots is summed to 1.
        for onehot in onehots:
            self.assertEqual(np.sum(onehot), 1)


class PongEnvironmentCase(AbstractPongEnvironmentCase, unittest.TestCase):
    ENV = PongEnvironment
    AGENT = MockPongAgent


class VectorizedPongEnvironmentCase(AbstractPongEnvironmentCase, unittest.TestCase):
    ENV = VectorizedPongEnvironment
    AGENT = MockPongAgent


if __name__ == "__main__":
    unittest.main()
