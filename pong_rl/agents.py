import abc

import numpy as np
from scipy.special import softmax
from tensorflow import keras


class BaseAgent(abc.ABC):
    """ Base Abstract class for trainable agent. """

    def __init__(self, actions_len, observation_shape):
        """ Agent initialization. """
        self.actions_len = actions_len
        self.observation_shape = observation_shape
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


class BaseAgentML(BaseAgent):
    """ Abstract Base Class for ML model-based agents. """

    def __init__(self, *args, **kwargs):
        """ Agent initialization is model creation. """
        super().__init__(*args)
        self._model = self._create_model(**kwargs)

    def _create_model(self, **kwargs):
        """ Model creation interface method. """
        self._model = self._create_model_impl(**kwargs)
        return self._model

    @abc.abstractmethod
    def _create_model_impl(self, **kwargs):
        """ Method should initialize model and return it. """
        pass

    def _predict_impl(self, observations):
        """ Agent prediction using ML model. """
        return self._model(self._prepare_observations(observations)).numpy()

    def _train_impl(self, observations, actions, rewards, **kwargs):
        """ Training ML model with given observations. """
        return self._model.fit(
            self._prepare_observations(observations), actions, sample_weight=rewards, **kwargs
        )

    def _prepare_observations(self, observations):
        """ Preprocess observations for prediction or training if needed. """
        return observations


class AgentRandom(BaseAgent):
    """ Agent that implements random acton choice. """

    def _predict_impl(self, observations):
        """ Predictions is made randomly. """
        observations_num = len(observations)
        return softmax(np.random.randn(observations_num, self.actions_len), axis=1)

    def _train_impl(self, observations, actions, rewards, **kwargs):
        """ Random Agent is unable to train. """
        return None

    @property
    def summary(self):
        """ Agent description. """
        return "AgentRandom"


class AgentKerasConv(BaseAgentML):
    """ Agent based on Keras (TensorFlow) Convolutional Neural Network. """

    def _create_model_impl(self, learning_rate=1e-3):
        """ Creating and compiling TF (Keras) model. """
        model = keras.Sequential(
            [
                keras.Input(shape=(self.observation_shape + (1,))),
                keras.layers.Conv2D(8, 8),
                keras.layers.MaxPool2D(4),
                keras.layers.Conv2D(16, 4),
                keras.layers.MaxPool2D(2),
                keras.layers.Conv2D(4, 2),
                keras.layers.MaxPool2D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.actions_len, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[
                keras.metrics.Accuracy(),
                keras.metrics.CategoricalCrossentropy(),
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.CategoricalHinge(),
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.TruePositives(),
                keras.metrics.TrueNegatives(),
                keras.metrics.AUC(),
            ],
        )
        return model

    def _prepare_observations(self, observations):
        """ Reshaping observations for convolutional layers. """
        return observations.view().reshape(observations.shape + (1,))

    @property
    def summary(self):
        """ Agent summary description. """
        summary_lines = []
        self._model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)
