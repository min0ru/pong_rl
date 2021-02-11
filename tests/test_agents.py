import abc
import numpy as np
import unittest

from pong_rl.environments import PongEnvironment
from pong_rl.agents import PongAgentRandom, PongAgentTF


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
        self.agent.train(observations, actions, rewards, verbose=False)
        self.assertGreater(self.agent.trainings, 0)


class RandomAgentCase(AbstractAgentCase, unittest.TestCase):
    """ Test random agent. """
    ENV = PongEnvironment
    AGENT = PongAgentRandom


class TFAgentCase(AbstractAgentCase, unittest.TestCase):
    """ Test Tensorflow agent. """
    ENV = PongEnvironment
    AGENT = PongAgentTF

    @staticmethod
    def _weights_equal(w1, w2):
        """ Compare two TF model weight lists. """
        return np.all([np.array_equal(a1, a2) for a1, a2 in zip(w1, w2)])

    def test_agent_train_model_weights_changed(self):
        """ Test that actual TF model weights are changing during agent training. """
        # Check that weights does not change without training

        # Save current model weights
        model = self.agent._model
        weights_before_train = model.get_weights()

        # Check that weights are not changed before model training.
        self.assertTrue(
            self._weights_equal(weights_before_train, model.get_weights()),
            'Model was not trained yet. Saved and actual weights should be equal.'
        )

        # # Play episode, train agent on episode observations
        observations, actions, rewards, score = self.env.play_episode(self.agent)
        self.agent.train(observations, actions, rewards, verbose=False)

        weights_after_train = model.get_weights()
        self.assertFalse(
            self._weights_equal(weights_before_train, weights_after_train),
            'Model weights should change after agent training.'
        )


if __name__ == '__main__':
    unittest.main()
