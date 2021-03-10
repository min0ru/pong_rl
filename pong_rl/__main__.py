from multiprocessing import Pipe, Process
from pathlib import Path
from operator import itemgetter

import numpy as np

from pong_rl.agents import PongAgentRandom, PongAgentTF
from pong_rl.environments import PongEnvironment, VectorizedPongEnvironment
from pong_rl.timer import ContextTimer

SAVED_MODEL = "data/convolution_v1/"


def renderer(pipe, environment, saved_model):
    """ Receive actual agent from pipe and render it in provided environment. """
    print("[render_process]: Render process started")
    agent = PongAgentTF()

    print("[render_process]: Awaiting for new agent data")
    episode = pipe.recv()
    print("[render_process]: Received agent update signal")

    if Path(saved_model).exists():
        agent._model.load_weights(saved_model)
        print("[render_process]: Loaded Weights")
    else:
        print("[render_process]: Model loading failed")

    print("[render_process]: Starting episodes rendering")

    should_run = True
    while should_run:
        environment.play_episode(agent, render=True)
        if pipe.poll():
            episode = pipe.recv()
            if Path(saved_model).exists():
                agent._model.load_weights(saved_model)
                print(f"[render_process]: Received and updated new agent from episode {episode}")
            else:
                print("[render_process]: Model loading failed")


def main():
    np.set_printoptions(precision=4, floatmode="maxprec", edgeitems=16, linewidth=120)

    pong = VectorizedPongEnvironment(num_environments=100)
    pong_render = PongEnvironment()
    saved_model = SAVED_MODEL

    print("Starting rendering process")
    child_pipe, parent_pipe = Pipe()
    render_process = Process(target=renderer, args=(child_pipe, pong_render, saved_model))
    render_process.start()

    agent_tf = PongAgentTF()
    agent_rnd = PongAgentRandom()
    agent = agent_tf

    if Path(saved_model).exists():
        print("Loading saved model weights")
        agent_tf._model.load_weights(saved_model)
    else:
        print("Cannot find model data in path:", Path(saved_model).absolute())

    print(agent_tf.summary)

    episode = 0
    should_train = True
    while should_train:
        print(f"Starting [{episode}] episode")
        with ContextTimer("Episode Timer"):
            # if (episode + 1) % 2 == 0:
            #     agent = agent_rnd
            #     print('Switching to *random agent* for current episode!')
            # else:
            #     agent = agent_tf
            ep_observations, ep_actions, ep_rewards, ep_score = pong.play_episode(
                agent, render=False
            )

        # print('Filtering only positive rewards')
        # positive = ep_rewards > 0
        # ep_observations = ep_observations[positive]
        # ep_actions = ep_actions[positive]
        # ep_rewards = ep_rewards[positive]

        print(f"Episode [{episode}] observations number: {len(ep_observations)}")
        print(f"Episode [{episode}] score: {ep_score.astype(np.int)}")
        print(f"Episode [{episode}] average score: {np.average(ep_score)}")
        print(f"Episode [{episode}] max score: {np.max(ep_score)}")

        # print("Actions:\n", ep_actions)
        unique_actions, actions_num = np.unique(ep_actions, axis=0, return_counts=True)
        unique_actions = [list(a) for a in list(unique_actions.astype(np.int))]
        actions_stats = sorted(zip(unique_actions, actions_num), key=itemgetter(0))
        print("Actions statistics:", actions_stats)
        print("Rewards:\n", ep_rewards)

        if len(ep_observations) > 0:
            with ContextTimer("Training Timer"):
                agent_tf.train(
                    ep_observations,
                    ep_actions,
                    ep_rewards,
                    batch_size=1024,
                )
        else:
            print("No training data available, skip training")

        print("Saving model weights")
        if not Path(saved_model).exists():
            Path(saved_model).mkdir()
        agent_tf._model.save_weights(saved_model)

        # Updating rendering agent
        print("Updating rendering agent")
        parent_pipe.send(episode)

        episode += 1

    render_process.join()


if __name__ == "__main__":
    main()
