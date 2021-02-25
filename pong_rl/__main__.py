from multiprocessing import Process, Pipe
from pathlib import Path
import pickle

from pong_rl.agents import PongAgentTF, PongAgentRandom
from pong_rl.environments import PongEnvironment, VectorizedPongEnvironment
from pong_rl.timer import ContextTimer


SAVED_MODEL = 'data/model.dat'


def renderer(pipe, environment, saved_model):
    """ Receive actual agent from pipe and render it in provided environment. """
    print('[render_process]: Render process started')
    agent = PongAgentTF()

    print('[render_process]: Awaiting for new agent data')
    episode = pipe.recv()
    print('[render_process]: Received agent update signal')

    if Path(saved_model).exists():
        agent._model.load_weights(saved_model)
        print('[render_process]: Loaded Weights')

    print('[render_process]: Starting episodes rendering')

    should_run = True
    while should_run:
        environment.play_episode(agent, render=True)
        if pipe.poll():
            episode = pipe.recv()
            if Path(saved_model).exists():
                agent._model.load_weights(saved_model)
            print(f'[render_process] Received and updated new agent from episode {episode}')


def main():
    pong = VectorizedPongEnvironment(num_environments=128)
    pong_render = PongEnvironment()

    saved_model = SAVED_MODEL
    if Path(saved_model).exists():
        print('Loading saved model weights')
        agent_tf._model.load_weights(saved_model)

    print('Starting rendering process')
    child_pipe, parent_pipe = Pipe()
    render_process = Process(target=renderer, args=(child_pipe, pong_render, saved_model))
    render_process.start()

    agent_tf = PongAgentTF()
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

        # Updating rendering agent
        print('Updating rendering agent')
        parent_pipe.send(episode)


    render_process.join()


if __name__ == '__main__':
    main()
