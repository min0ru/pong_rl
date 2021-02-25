import numpy as np
from pathlib import Path

from pong_rl.agents import PongAgentTF, PongAgentRandom
from pong_rl.environments import PongEnvironment, VectorizedPongEnvironment
from pong_rl.timer import ContextTimer


def main():
    pong = VectorizedPongEnvironment(num_environments=32)
    pong_render = PongEnvironment()
    agent_tf = PongAgentTF()

    saved_model = 'data/model.dat'
    if Path(saved_model).exists():
        print('Loading saved model weights')
        agent_tf._model.load_weights(saved_model)

    print(agent_tf.summary)

    episode = 0
    should_train = True
    while should_train:
        print(f"Starting [{episode}] episode")
        with ContextTimer("Episode Timer"):
            ep_observations, ep_actions, ep_rewards, ep_score = \
                pong.play_episode(agent_tf, render=False)

        print(f'Episode [{episode}] observations number: {len(ep_observations)}')
        print(f'Episode [{episode}] score: {ep_score}')

        print('Actions:\n', ep_actions)
        print('Rewards:\n', ep_rewards)

        with ContextTimer("Training Timer"):
            agent_tf.train(
                ep_observations,
                ep_actions,
                ep_rewards,
                batch_size=2048,
            )
        episode += 1

        print('Saving model weights')
        agent_tf._model.save_weights('data/model.dat')

        # if episode % 5 == 0:
        print('Playing renderable demo episode')
        for _ in range(4):
            pong_render.play_episode(agent_tf, render=True)
        print('Finished demo episode')


if __name__ == '__main__':
    main()
