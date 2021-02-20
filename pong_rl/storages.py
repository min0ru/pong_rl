

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