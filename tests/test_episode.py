import abc
import copy
import numpy as np
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
        # Check that weights does not change without training
        weights_before_train = copy.deepcopy(self.agent._model.trainable_weights)
        for w1, w2 in zip(weights_before_train, self.agent._model.trainable_weights):
            self.assertTrue(
                np.all(np.equal(w1, w2)),
                'Saved weights changed before training!'
            )

        # Play episode, train agent on episode observations
        observations, actions, rewards, score = self.env.play_episode(self.agent)
        self.agent.train(observations, actions, rewards)

        # Check that model weights was changed after training
        weights_after_train = copy.deepcopy(self.agent._model.trainable_weights)
        for w1, w2 in zip(weights_before_train, weights_after_train):
            self.assertFalse(
                np.all(np.equal(w1, w2)),
                'Network weights did not changed after training!'
            )


if __name__ == '__main__':
    unittest.main()
