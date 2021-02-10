import unittest

import numpy as np

from pong_rl.environment import PongEnvironment


class TestPongEnvironment(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PongEnvironment()

    def test_process_episode_rewards(self):
        gamma = 0.99
        testing_rewards = [
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



if __name__ == '__main__':
    unittest.main()
