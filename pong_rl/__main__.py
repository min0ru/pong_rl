import logging
import sys
from multiprocessing import Pipe, Process
from operator import itemgetter
from pathlib import Path

import numpy as np

from pong_rl.agents import AgentKerasConv
from pong_rl.environments import PongEnvironment, VectorizedPongEnvironment
from pong_rl.timer import ContextTimer

# TODO: Set model name/version as parameter
MODEL_NAME = "convolution_v6"
MODEL_FILE = f"{MODEL_NAME}.h5"
EPISODE_FILE = f"{MODEL_NAME}.episode"
SAVED_MODEL = Path("data", MODEL_FILE)
SAVED_EPISODE = Path("data", EPISODE_FILE)
MAX_OBSERVATIONS = 110000


def get_logger(name, level=logging.INFO):
    """ Create logger for main process. """
    logger = logging.getLogger(name)
    log_format = "[%(asctime)s] %(message)s"
    date_format = "%d.%m.%Y %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    file_handler = logging.FileHandler(Path("log", f"{name}.log"))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.setLevel(level)

    return logger


def renderer(pipe, environment, saved_model, agent_class):
    """ Receive actual agent from pipe and render it in provided environment. """
    print("[render_process]: Render process started")
    agent = agent_class(environment.actions_len, environment.observation_shape)

    print("[render_process]: Awaiting for new agent data")
    _ = pipe.recv()
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


# TODO: Extract AgentTrainingStrategy class with custom training logic and tests
def main():
    np.set_printoptions(precision=4, floatmode="maxprec", edgeitems=16, linewidth=1200)

    log = get_logger(MODEL_NAME, logging.INFO)

    saved_model = SAVED_MODEL
    saved_episode = SAVED_EPISODE

    # TODO: Set training parameters as command-line arguments
    # Training hyperparameters
    agent_class = AgentKerasConv
    learning_rate = 1e-6
    positive_feedback_multiplier = 5
    games_per_episode = 64

    log.info("Training hyperparameters:")
    log.info(f"agent_class: {agent_class}")
    log.info(f"learning_rate: {learning_rate}")
    log.info(f"positive_feedback_multiplier: {positive_feedback_multiplier}")
    log.info(f"games_per_episode: {games_per_episode}")

    # Environments
    pong_render = PongEnvironment()
    pong = VectorizedPongEnvironment(num_environments=games_per_episode)

    log.info("Starting rendering process")
    child_pipe, parent_pipe = Pipe()
    render_process = Process(
        target=renderer, args=(child_pipe, pong_render, saved_model, agent_class)
    )
    render_process.start()

    agent = agent_class(pong.actions_len, pong.observation_shape, learning_rate=learning_rate)

    if saved_model.exists():
        log.info("Loading saved model weights")
        agent._model.load_weights(saved_model)
    else:
        log.info(f"Cannot find model data in path: {saved_model.absolute()}")

    log.info(f"Agent summary:\n{agent.summary}")

    if saved_episode.exists():
        episode = int(saved_episode.read_text())
    else:
        episode = 0

    should_train = True
    while should_train:
        log.info(f"Starting [{episode}] episode")
        with ContextTimer("Episode Timer", log):
            ep_observations, ep_actions, ep_rewards, ep_score = pong.play_episode(
                agent, render=False
            )

        # Balancing positive / negative rewards
        positive_rewards = ep_rewards > 0
        positive_rewards_num = len(ep_rewards[positive_rewards])
        negative_rewards = ep_rewards <= 0
        negative_rewards_num = len(ep_rewards[negative_rewards])
        try:
            rewards_ratio = negative_rewards_num / positive_rewards_num
        except ZeroDivisionError:
            rewards_ratio = 1
        log.info(f"Episode [{episode}] rewards len positive: {positive_rewards_num}")
        log.info(f"Episode [{episode}] rewards len negative: {negative_rewards_num}")
        log.info(f"Rebalancing rewards with rewards_ratio: {rewards_ratio}")
        log.info(f"Positive feedback multiplier is: {positive_feedback_multiplier}")
        ep_rewards[positive_rewards] *= rewards_ratio * positive_feedback_multiplier

        log.info(f"Episode [{episode}] observations number: {len(ep_observations)}")
        log.info(f"Episode [{episode}] score: {ep_score.astype(int)}")
        log.info(f"Episode [{episode}] average score: {np.average(ep_score)}")
        log.info(f"Episode [{episode}] max score: {np.max(ep_score)}")

        unique_actions, actions_num = np.unique(ep_actions, axis=0, return_counts=True)
        unique_actions = [list(a) for a in list(unique_actions.astype(int))]
        actions_percent = np.rint(actions_num / np.sum(actions_num) * 100).astype(int)
        actions_stats = sorted(zip(unique_actions, actions_num, actions_percent), key=itemgetter(0))
        log.info(f"Actions statistics: {actions_stats}")
        log.info(f"Rewards:\n {ep_rewards}")

        if len(ep_observations) > 0:
            with ContextTimer("Training Timer", log):
                train_metrics = agent.train(
                    ep_observations,
                    ep_actions,
                    ep_rewards,
                    batch_size=1024,
                )
                log.info(f"Episode {episode} train metrics: {train_metrics.history}")
        else:
            log.info("No training data available, skip training")

        if len(ep_observations) > MAX_OBSERVATIONS:
            games_per_episode -= 1
            log.info(
                f"Episode [{episode}] number of observations exceeding {MAX_OBSERVATIONS}, "
                f"reducing games_per_episode to {games_per_episode} and rebuilding environment."
            )
            pong = VectorizedPongEnvironment(num_environments=games_per_episode)

        log.info("Saving model weights")
        agent._model.save_weights(saved_model)
        saved_episode.write_text(str(episode))

        log.info("Updating rendering agent")
        parent_pipe.send(episode)

        episode += 1

    render_process.join()


if __name__ == "__main__":
    main()
