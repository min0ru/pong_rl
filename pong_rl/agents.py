import abc
import numpy as np

from scipy.special import softmax
from tensorflow import keras

from .actions import PongAction


class BasePongAgent(abc.ABC):
    """ Base Abstract class for trainable Pong Agent. """
    INPUT_SHAPE = (80, 80)  # Processed input shape
    ACTIONS_LEN = len(PongAction)

    def __init__(self):
        """ Agent initialization. """
        self.trainings = 0
        self.trained_observations = 0

    def predict(self, observations):
        """ Agent actions prediction based on given observations. """
        predictions = self._predict_impl(observations)
        return predictions

    @abc.abstractmethod
    def _predict_impl(self, observations):
        """ Agent action prediction. Should be implemented in child class. """
        pass

    def train(self, observations, actions, rewards, **kwargs):
        """ Agent training (Reinforced Learning). """
        self.trainings += 1
        self.trained_observations += len(observations)
        return self._train_impl(observations, actions, rewards, **kwargs)

    @abc.abstractmethod
    def _train_impl(self, observations, actions, rewards, **kwargs):
        """ Agent training implementation. Should be implemented in child class. """
        pass


class PongAgentRandom(BasePongAgent):
    """ Pong Agent that implements random acton choice. """

    def _predict_impl(self, observations):
        """ Predictions is made randomly. """
        observations_num = len(observations)
        return softmax(np.random.randn(observations_num, self.ACTIONS_LEN), axis=1)

    def _train_impl(self, observations, actions, rewards, **kwargs):
        """ Random Agent is unable to train. """
        return None

    @property
    def summary(self):
        """ Agent description. """
        return 'RandomAgent'


class PongAgentTF(BasePongAgent):
    """ Pong Agent based on TensorFlow NeuralNetworks. """

    def __init__(self):
        """ Initialize agent by creating TF model. """
        super().__init__()

        self._model = keras.Sequential([
            keras.Input(shape=(self.INPUT_SHAPE + (1, ))),
            keras.layers.Conv2D(32, 4),
            keras.layers.MaxPool2D(4),
            keras.layers.Conv2D(16, 4),
            keras.layers.MaxPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.ACTIONS_LEN, activation='softmax')
        ])
        self._model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=1e-6),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def _prepare_observations(self, observations):
        return observations.view().reshape(observations.shape + (1, ))

    def _predict_impl(self, observations):
        """ Agent prediction using TensorFlow NN model. """
        return self._model(self._prepare_observations(observations)).numpy()

    def _train_impl(self, observations, actions, rewards, **kwargs):
        """ Training NN model with given observations. """
        return self._model.fit(
            self._prepare_observations(observations),
            actions,
            sample_weight=rewards,
            **kwargs
        )

    @property
    def summary(self):
        """ Agent description. """
        return self._model.summary()
