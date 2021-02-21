

class MultiList:
    """ MultiList is a list of lists of the same length. """
    def __init__(self, length):
        if length <= 0:
            raise TypeError('Number of sublists should be more than one!')
        self.length = length
        self.list = [list() for _ in range(length)]

    def append(self, iterable):
        """ Append new elements from a given iterable by spreading it over sublists. """
        if len(iterable) != self.length:
            raise TypeError('Appended sublist should be the same length as MultiList!')
        for stored_sublist, new_element in zip(self.list, iterable):
            stored_sublist.append(new_element)

    def flat(self):
        """ Return flat list by uniting sublists. """
        return sum(self.list, [])

    def __iter__(self):
        """ MultiList could be iterated as list of sublists. """
        return iter(self.list)

    def __len__(self):
        """ MultiList length is a number of sublists. """
        return len(self.list)


class EpisodeStorage:
    """ Storage for output data of a multiple environments.

    Storage accepts and stores output from multiple environments and outputs stored data
    as stacked sequence of observations as if it was output of just one environment.
    """

    def __init__(self, num_environments):
        self.num_environments = num_environments

    def add(self, observations, rewards, dones, infos):
        pass

    def get(self):
        pass