import abc
import numpy as np

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

    def predict(self, observation):
        """ Agent action prediction based on given observation. """
        prediction = self._predict_impl(observation)
        return prediction

    @abc.abstractmethod
    def _predict_impl(self, observation):
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

    @staticmethod
    def _softmax(x):
        """ Helper function for random predictions to sum up to 1. """
        return np.exp(x) / sum(np.exp(x))

    def _predict_impl(self, observation):
        """ Prediction is made randomly. """
        return self._softmax(np.random.randn(self.ACTIONS_LEN))

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
            keras.Input(shape=self.INPUT_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.ACTIONS_LEN, activation='softmax')
        ])
        self._model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=1e-6),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def _predict_impl(self, observation):
        """ Agent prediction using TensorFlow NN model. """
        input_batch = observation[np.newaxis]
        return self._model(input_batch)[0].numpy()

    def _train_impl(self, observations, actions, rewards, **kwargs):
        """ Training NN model with given observations. """
        return self._model.fit(
            observations,
            actions,
            sample_weight=rewards,
            **kwargs
        )

    @property
    def summary(self):
        """ Agent description. """
        return self._model.summary()
