from pong_rl.agent import PongAgentRandom, PongAgentTF
from pong_rl.environment import PongEnvironment
from pong_rl.timer import ContextTimer

import numpy as np


def size_mb(nparray):
    return round(nparray.nbytes / 1024 / 1024, 2)


def main():
    pong = PongEnvironment()
    # agent_rnd = PongAgentRandom()
    agent_tf = PongAgentTF()

    print(agent_tf.summary)
    with ContextTimer('Total Time (PongAgentTF) WARMUP!'):
        observations, actions, rewards = np.array([]), np.array([]), np.array([])
        episode_num = 64
        for episode in range(episode_num):
            with ContextTimer(f'Episode {episode}'):
                ep_observations, ep_actions, ep_rewards, ep_score = \
                    pong.play_episode(agent_tf, render=False)

                print('Episode observations number:', len(ep_observations))
                print('Episode score:', ep_score)

                if observations.any():
                    observations = np.append(observations, ep_observations, axis=0)
                    actions = np.append(actions, ep_actions, axis=0)
                    rewards = np.append(rewards, ep_rewards, axis=0)
                else:
                    observations = ep_observations
                    actions = ep_actions
                    rewards = ep_rewards

        print('')
        print(f'Total observations length {len(observations)} after {episode_num} episodes')
        print(f'Observation shape: {observations.shape} size: {size_mb(observations)} MB')
        print(f'Actions shape: {actions.shape} size: {size_mb(actions)} MB')
        print(f'Rewards shape: {rewards.shape} size: {size_mb(rewards)} MB')

    print('Warmup finished! Training!')
    agent_tf.train(observations, actions, rewards)

    for episode in range(256):
        ep_observations, ep_actions, ep_rewards, ep_score = \
            pong.play_episode(agent_tf, render=True)

        print('Episode observations number:', len(ep_observations))
        print(f'Episode [{episode}] score:', ep_score)

        print('Actions:\n', ep_actions)
        # print('Rewards:\n', ep_rewards)

        agent_tf.train(ep_observations, ep_actions, ep_rewards)


if __name__ == '__main__':
    main()
