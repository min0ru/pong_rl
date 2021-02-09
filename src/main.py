from agents import PongAgentRandom, PongAgentTF
from environments import PongEnvironment
from timers import ContextTimer


def main():
    pong = PongEnvironment()
    agent_tf = PongAgentTF()
    agent_rnd = PongAgentRandom()

    print(agent_rnd.summary)
    with ContextTimer('Total Time (PongAgentRandom)'):
        for episode in range(10):
            with ContextTimer(f'Episode {episode}'):
                # observations, actions, rewards = pong.play_episode(agent_rnd, render=False)
                pong.play_episode(agent_rnd, render=True)

    print(agent_tf.summary)
    with ContextTimer('Total Time (PongAgentTF)'):
        for episode in range(10):
            with ContextTimer(f'Episode {episode}'):
                # observations, actions, rewards = pong.play_episode(agent_tf, render=False)
                pong.play_episode(agent_tf, render=True)


if __name__ == '__main__':
    main()
