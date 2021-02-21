from itertools import groupby
import numpy as np
import unittest

from pong_rl.storages import MultiList, EpisodeStorage


class MultiListCase(unittest.TestCase):
    def test_sublists_num_less_than_one(self):
        """ MultiList should be initialized with correct number of sublists. """
        self.assertRaises(TypeError, MultiList, -1)

    def test_sublist_len(self):
        """ MultiList should return number of sublists as it's length. """
        num_sublists = 10
        mlist = MultiList(num_sublists)
        self.assertEqual(len(mlist), num_sublists)

    def test_append_and_flat(self):
        num_sublists = 10
        mlist = MultiList(num_sublists)

        # Fill MultiList with lists like [0, 1, 2, 3, 4, 5]
        # So all resulting sublist will contain identical numbers [[0, 0, 0], [1, 1, 1] ...]]
        num_iterations = 10
        for _ in range(num_iterations):
            sublist = list(range(len(mlist)))
            mlist.append(sublist)

        # Output should be flat list which consists of stacked elements of all sublists.
        flat_list = mlist.flat()

        # Check that total number of elements in MultiList correspond with inserted.
        total_element_length = sum(len(sublist) for sublist in mlist)
        self.assertEqual(total_element_length, num_sublists * num_iterations)
        self.assertEqual(len(flat_list), num_sublists * num_iterations)

        # Check that flat list outputs inserted numbers in the right sequence.
        for (key, group), number in zip(groupby(flat_list), range(num_sublists)):
            self.assertEqual(key, number)
            self.assertEqual(len(list(group)), num_sublists)


class EpisodeStorageCase(unittest.TestCase):
    def test_multiple_environments(self):
        """ Test vectorized storage with multiple environment inputs with random sample data. """
        for num_environments in [1, 2, 3, 16]:
            storage = EpisodeStorage(num_environments)
            num_frames = 10
            total_frames = num_frames * num_environments

            for _ in range(num_frames):
                observations = np.random.randint(256, size=(num_environments, 80, 80))
                rewards = np.random.randn(num_environments)
                dones = [False] * num_environments
                infos = [{'empty': True}] * num_environments

                storage.add(observations, rewards, dones, infos)
            # Retrieve episodes data from storage
            observations, rewards, dones, infos = storage.get()

            # Check that storage returns all inserted data
            self.assertEqual(len(observations), total_frames)
            self.assertEqual(len(rewards), total_frames)
            self.assertEqual(len(dones), total_frames)
            self.assertEqual(len(infos), total_frames)

    def test_storage_order(self):
        """ Check that storage returns observations and other elements in right order. """
        storage = EpisodeStorage(3)

        # Insert data from 3 fake environments
        storage.add([1, 2, 3], [5, 6, 7], [8, 9, 10], [11, 12, 13])
        storage.add([1, 2, 3], [5, 6, 7], [8, 9, 10], [11, 12, 13])
        storage.add([1, 2, 3], [5, 6, 7], [8, 9, 10], [11, 12, 13])

        observations, rewards, dones, infos = storage.get()

        # Check that environment data is stacked in sequential way and right order.
        self.assertListEqual(observations, [1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.assertListEqual(rewards, [5, 5, 5, 6, 6, 6, 7, 7, 7])
        self.assertListEqual(dones, [8, 8, 8, 9, 9, 9, 10, 10, 10])
        self.assertListEqual(infos, [11, 11, 11, 12, 12, 12, 13, 13, 13])


if __name__ == '__main__':
    unittest.main()
