import logging
import sys
from multiprocessing import Pipe, Process
from operator import itemgetter
from pathlib import Path

import numpy as np

from pong_rl.agents import PongAgentRandom, PongAgentTF
from pong_rl.environments import PongEnvironment, VectorizedPongEnvironment
from pong_rl.timer import ContextTimer

MODEL_NAME = "convolution_v1"
MODEL_FILE = f"{MODEL_NAME}.h5"
EPISODE_FILE = f"{MODEL_NAME}.episode"
SAVED_MODEL = Path("data", MODEL_FILE)
SAVED_EPISODE = Path("data", EPISODE_FILE)


def get_logger(name, level=logging.INFO):
    """ Create logger for main process. """
    logger = logging.getLogger(name)
    log_format = "[%(asctime)s] %(message)s"
    date_format = "%d.%m.%Y %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    file_handler = logging.FileHandler(Path("log", name))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.setLevel(level)

    return logger


def renderer(pipe, environment, saved_model):
    """ Receive actual agent from pipe and render it in provided environment. """
    print("[render_process]: Render process started")
    agent = PongAgentTF()

    print("[render_process]: Awaiting for new agent data")
    episode = pipe.recv()
    print("[render_process]: Received agent update signal")

    if saved_model.exists():
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
            if saved_model.exists():
                agent._model.load_weights(saved_model)
                print(f"[render_process]: Received and updated new agent from episode {episode}")
            else:
                print("[render_process]: Model loading failed")


def main():
    np.set_printoptions(precision=4, floatmode="maxprec", edgeitems=16, linewidth=1200)

    log = get_logger(MODEL_NAME, logging.INFO)

    pong = VectorizedPongEnvironment(num_environments=100)
    pong_render = PongEnvironment()
    saved_model = SAVED_MODEL
    saved_episode = SAVED_EPISODE

    log.info("Starting rendering process")
    child_pipe, parent_pipe = Pipe()
    render_process = Process(target=renderer, args=(child_pipe, pong_render, saved_model))
    render_process.start()

    agent_tf = PongAgentTF()
    # agent_rnd = PongAgentRandom()
    agent = agent_tf

    if saved_model.exists():
        log.info("Loading saved model weights")
        agent_tf._model.load_weights(saved_model)
    else:
        log.info(f"Cannot find model data in path: {saved_model.absolute()}")

    agent_tf.summary

    if saved_episode.exists():
        episode = int(saved_episode.read_text())
    else:
        episode = 0

    should_train = True
    while should_train:
        log.info(f"Starting [{episode}] episode")
        with ContextTimer("Episode Timer", log):
            # if (episode + 1) % 2 == 0:
            #     agent = agent_rnd
            #     log.info("Switching to *random agent* for current episode!")
            # else:
            #     agent = agent_tf
            ep_observations, ep_actions, ep_rewards, ep_score = pong.play_episode(
                agent, render=False
            )

        # log.info("Filtering only positive rewards")
        # positive = ep_rewards > 0
        # ep_observations = ep_observations[positive]
        # ep_actions = ep_actions[positive]
        # ep_rewards = ep_rewards[positive]

        positive_rewards = ep_rewards >= 0
        positive_rewards_num = len(ep_rewards[positive_rewards])
        negative_rewards = ep_rewards < 0
        negative_rewards_num = len(ep_rewards[negative_rewards])
        rewards_ratio = negative_rewards_num / positive_rewards_num
        log.info(f"Episode [{episode}] rewards len positive: {positive_rewards_num}")
        log.info(f"Episode [{episode}] rewards len negative: {negative_rewards_num}")
        if positive_rewards_num < negative_rewards_num:
            log.info(f"Rebalancing rewards with positive/negative ratio is {rewards_ratio}")
            ep_rewards[positive_rewards] *= rewards_ratio

        log.info(f"Episode [{episode}] observations number: {len(ep_observations)}")
        log.info(f"Episode [{episode}] score: {ep_score.astype(np.int)}")
        log.info(f"Episode [{episode}] average score: {np.average(ep_score)}")
        log.info(f"Episode [{episode}] max score: {np.max(ep_score)}")

        # log.info(f"Actions:\n" {ep_actions}")
        unique_actions, actions_num = np.unique(ep_actions, axis=0, return_counts=True)
        unique_actions = [list(a) for a in list(unique_actions.astype(np.int))]
        actions_percent = np.rint(actions_num / np.sum(actions_num) * 100).astype(np.int)
        actions_stats = sorted(zip(unique_actions, actions_num, actions_percent), key=itemgetter(0))
        log.info(f"Actions statistics: {actions_stats}")
        log.info(f"Rewards:\n {ep_rewards}")

        if len(ep_observations) > 0:
            with ContextTimer("Training Timer", log):
                train_metrics = agent_tf.train(
                    ep_observations,
                    ep_actions,
                    ep_rewards,
                    batch_size=1024,
                )
                log.info(f"Episode {episode} train metrics: {train_metrics.history}")
        else:
            log.info("No training data available, skip training")

        log.info("Saving model weights")
        agent_tf._model.save_weights(saved_model)
        saved_episode.write_text(str(episode))

        # Updating rendering agent
        log.info("Updating rendering agent")
        parent_pipe.send(episode)

        episode += 1

    render_process.join()


if __name__ == "__main__":
    main()
