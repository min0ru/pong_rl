import abc
import unittest

from pong_rl.environment import PongEnvironment
from pong_rl.agent import PongAgentRandom, PongAgentTF


class AbstractAgentCase(abc.ABC):
    """ ABC class for agents training. """
    ENV = None
    AGENT = None

    def setUp(self):
        """ Initialize environment and agent. """
        self.env = self.ENV()
        self.agent = self.AGENT()

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

    def test_agent_train(self):
        """ Test agent training with episode results. """
        observations, actions, rewards, score = self.env.play_episode(self.agent)
        self.agent.train(observations, actions, rewards)
        self.assertGreater(self.agent.trainings, 0)


class RandomAgentCase(AbstractAgentCase, unittest.TestCase):
    """ Test random agent. """
    ENV = PongEnvironment
    AGENT = PongAgentRandom


class TFAgentCase(AbstractAgentCase, unittest.TestCase):
    """ Test Tensorflow agent. """
    ENV = PongEnvironment
    AGENT = PongAgentTF

    def test_agent_train_model_weights_changed(self):
        """ Test that actual TF model weights are changing during agent training. """
        weights_before_train = self.agent._model.trainable_weights.copy()

        # Check that weights does not change without training
        self.assertEqual(weights_before_train, self.agent._model.trainable_weights)

        # Play episode, train agent on episode observations
        observations, actions, rewards, score = self.env.play_episode(self.agent)
        self.agent.train(observations, actions, rewards)

        # Check that model weights was changed after training
        weights_after_train = self.agent._model.trainable_weights.copy()
        self.assertNotEqual(weights_before_train, weights_after_train)


if __name__ == '__main__':
    unittest.main()
