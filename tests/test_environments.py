import abc
import unittest

import numpy as np

from pong_rl.agents import PongAgentRandom
from pong_rl.environments import PongEnvironment


class MockPongAgent(PongAgentRandom):
    """ Mock Pong Agent that implements random acton choice. """

    @property
    def summary(self):
        """ Agent description. """
        return 'RandomPongAgent as Mock Agent'


class AbstractPongEnvironmentCase(abc.ABC):
    ENV = None
    AGENT = None

    def setUp(self) -> None:
        """ Initialize Environment. """
        self.env = self.ENV()
        self.agent = self.AGENT()

    def test_process_episode_rewards(self):
        """ Test episode rewards smoothing method for Pong. """
        gamma = 0.99
        testing_rewards = [
            (
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ),
            (
                [1, 0, 0, 0, 0],
                [1.0, 0, 0, 0, 0]
            ),
            (
                [0, 0, 0, 0, -1],
                [-0.96059601, -0.970299, -0.9801, -0.99, -1.]
            ),
            (
                [0, 0, 0, 0, 1],
                [0.96059601, 0.970299, 0.9801, 0.99, 1.]
            ),
            (
                [0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
                [0.96059601, 0.970299, 0.9801, 0.99, 1.,
                 -0.96059601, -0.970299, -0.9801, -0.99, -1.]
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

        self.assertEqual(observations_num, actions_num)
        self.assertEqual(observations_num, rewards_num)
        self.assertGreater(observations_num, 0)
        self.assertNotEqual(score, 0)


class PongEnvironmentCase(AbstractPongEnvironmentCase, unittest.TestCase):
    ENV = PongEnvironment
    AGENT = MockPongAgent


if __name__ == '__main__':
    unittest.main()
