import abc
import unittest

from pong_rl.environment import PongEnvironment
from pong_rl.agent import PongAgentRandom, PongAgentTF


class AbstractAgentCase(abc.ABC):
    ENV = None
    AGENT = None

    def setUp(self):
        self.env = self.ENV()
        self.agent = self.AGENT()

    def test_environment_output(self):
        """ Play one episode with RandomAgent and test that env output is not empty. """
        observations, actions, rewards, total_reward = self.env.play_episode(self.agent)
        observations_num = len(observations)
        actions_num = len(actions)
        rewards_num = len(rewards)

        self.assertEqual(observations_num, actions_num)
        self.assertEqual(observations_num, rewards_num)
        self.assertGreater(observations_num, 0)

        self.assertNotEqual(total_reward, 0)


class RandomAgentCase(AbstractAgentCase, unittest.TestCase):
    ENV = PongEnvironment
    AGENT = PongAgentRandom


class TFAgentCase(AbstractAgentCase, unittest.TestCase):
    ENV = PongEnvironment
    AGENT = PongAgentTF


if __name__ == '__main__':
    unittest.main()
