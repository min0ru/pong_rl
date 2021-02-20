import numpy as np
import unittest

from pong_rl.storages import EpisodeStorage


class EpisodeStorageCase(unittest.TestCase):
    def test_multiple_environments(self):
        """ Test vectorized storage with multiple environment inputs with random sample data. """
        for num_environments in [1, 2, 3, 16]:
            storage = EpisodeStorage(num_environments=num_environments)
            num_frames = 10
            total_frames = num_frames * num_environments

            observations_list = [list() for _ in range(num_environments)]

            for _ in range(num_frames):
                observations = np.random.randint(256, size=(num_environments, 80, 80))
                rewards = np.random.randn(num_environments)
                dones = [False] * num_environments
                infos = [{'empty': True}] * num_environments

                storage.add(observations, rewards, dones, infos)
                observations_list.append(observations[0])

            # Retrieve episodes data from storage
            observations, rewards, dones, infos = storage.get()

            # Check that storage returns all inserted data
            self.assertEqual(len(observations), total_frames)
            self.assertEqual(len(rewards), total_frames)
            self.assertEqual(len(dones), total_frames)
            self.assertEqual(len(infos), total_frames)

            # Check that storage output observations is exactly the same as plain list
            total_observations = sum([len(obs) for obs in observations])
            self.assertEqual(len(observations), total_observations)

            # Check that first num_frames observations is from the first environment output
            first_environment_observations = observations_list[0]
            np.testing.assert_array_equal(
                first_environment_observations,
                observations[:len(first_environment_observations)]
            )

            # Check that last num_frames observations is from the last environment output
            last_environment_observations = observations_list[-1]
            np.testing.assert_array_equal(
                last_environment_observations,
                observations[-len(last_environment_observations):]
            )


if __name__ == '__main__':
    unittest.main()
