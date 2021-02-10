import numpy as np

from tensorflow import keras


class BasePongAgent:
    INPUT_SHAPE = (80, 80)      # Processed input shape
    ACTIONS_LEN = 2

    def _process_observation(self, observation):
        """Extract area of interest 160x160 with down sampling and converting to b/w colors."""
        image = observation[34:194, :, :]
        image = image[::2, ::2, 1]
        image[image < 100] = 0
        image[image > 0] = 255
        image_batch = image[np.newaxis]
        return image_batch

    def predict(self, observation):
        processed_observation = self._process_observation(observation)
        prediction = self._predict_impl(processed_observation)
        return prediction

    def _predict_impl(self, processed_observation):
        raise NotImplementedError()


class PongAgentRandom(BasePongAgent):
    def _process_observation(self, observation):
        """ Random agent does not need to see anything. """
        return observation

    @staticmethod
    def _softmax(x):
        return np.exp(x) / sum(np.exp(x))

    def _predict_impl(self, processed_observation):
        return self._softmax(np.random.randn(self.ACTIONS_LEN))

    @property
    def summary(self):
        return 'RandomAgent'


class PongAgentTF(BasePongAgent):
    def __init__(self):
        super().__init__()
        self._model = keras.Sequential([
            keras.Input(shape=self.INPUT_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.ACTIONS_LEN, activation='softmax')
        ])
        self._model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def _predict_impl(self, processed_observation):
        return self._model(processed_observation)[0].numpy()

    @property
    def summary(self):
        return self._model.summary()
