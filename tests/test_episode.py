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
        observations, actions, rewards = self.env.play_episode(self.agent, render=False)
        self.assertTrue(len(observations) > 0)
        self.assertTrue(len(actions) > 0)
        self.assertTrue(len(rewards) > 0)


class RandomAgentCase(AbstractAgentCase, unittest.TestCase):
    ENV = PongEnvironment
    AGENT = PongAgentRandom


class TFAgentCase(AbstractAgentCase, unittest.TestCase):
    ENV = PongEnvironment
    AGENT = PongAgentTF


if __name__ == '__main__':
    unittest.main()
