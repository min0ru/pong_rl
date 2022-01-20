import sys
from pathlib import Path

from matplotlib import pyplot as plt
from pandas import DataFrame, Series


def parse_log(model_name, value_name):
    values = Series(
        float(line.split()[-1])
        for line in Path("log", f"{model_name}.log").open()
        if value_name in line
    )
    return values


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("model_name argument is required")
        print("rolling argument is required")
        exit(-1)

    model_name = sys.argv[1]
    rolling = int(sys.argv[2])

    avg_scores = parse_log(model_name, "average score")
    max_scores = parse_log(model_name, "max score")

    episode_stats = DataFrame(
        {
            "Average score": avg_scores,
            "Max score": max_scores,
        },
    )

    episode_stats.rolling(rolling).mean().plot()
    plt.title(model_name)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()
