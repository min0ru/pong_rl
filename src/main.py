import gym
import time

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


class PongEnv:
    ENV_NAME = 'Pong-v4'

    # TODO: convert actions to ENUM
    ACTION_NOP = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTIONS = [ACTION_UP, ACTION_DOWN]

    def __init__(self):
        self._env = gym.make(self.ENV_NAME)

    def play_episode(self, agent, render=False):
        """ Play episode using agent and return observations, actions, rewards. """
        episode_finished = False
        observations, actions, rewards = [], [], []
        observation = self._env.reset()
        while not episode_finished:
            if render:
                self._env.render()
            observations.append(observation)
            actions_probability = agent.predict(observation)
            actions.append(actions_probability)
            action = self._choose_probable_action(actions_probability)
            observation, reward, episode_finished, info = self._env.step(action)
            rewards.append(reward)
        return observations, actions, rewards

    def _choose_probable_action(self, probability):
        return np.random.choice(self.ACTIONS, p=probability)

    def _process_episode_rewards(self, rewards, gamma=0.99):
        """ Smooth reward for specific environment. """
        # TODO: gamma decay for Pong
        return rewards


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        print(f'[{self.name}] seconds: {self.total_time}')


def main():
    pong = PongEnv()
    agent_tf = PongAgentTF()
    agent_rnd = PongAgentRandom()

    print(agent_rnd.summary)
    with Timer('Total Time (PongAgentRandom)'):
        for episode in range(10):
            with Timer(f'Episode {episode}'):
                # observations, actions, rewards = pong.play_episode(agent_rnd, render=False)
                pong.play_episode(agent_rnd, render=True)

    print(agent_tf.summary)
    with Timer('Total Time (PongAgentTF)'):
        for episode in range(10):
            with Timer(f'Episode {episode}'):
                # observations, actions, rewards = pong.play_episode(agent_tf, render=False)
                pong.play_episode(agent_tf, render=True)


if __name__ == '__main__':
    main()
