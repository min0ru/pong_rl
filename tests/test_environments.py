import abc
import unittest

import numpy as np

from pong_rl.environments import PongEnvironment


class AbstractPongEnvironmentCase(abc.ABC):
    def setUp(self) -> None:
        """ Initialize Environment. """
        self.env = self.ENV()

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


class PongEnvironmentCase(AbstractPongEnvironmentCase, unittest.TestCase):
    ENV = PongEnvironment


if __name__ == '__main__':
    unittest.main()
