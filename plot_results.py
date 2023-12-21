import matplotlib.pyplot as plt
import pickle as pk
import numpy as np


def score_plot():
    with open("./Results/score_X.pkl", "rb") as f:
        ai_scores = np.array(pk.load(f))
    with open("./Results/score_Y.pkl", "rb") as f:
        random_scores = np.array(pk.load(f))

    num_plays = range(1, len(random_scores) + 1)
    plt.plot(num_plays, random_scores, label='Random Agent',
             color='blue', linestyle='-', marker='o', markersize=8)
    plt.plot(num_plays, ai_scores, label='AI Agent', color='red',
             linestyle='--', marker='^', markersize=8)
    plt.xlabel('Number of Plays', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Trend of Scores for Random Agent vs AI Agent', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='both', labelsize=10)
    plt.show()


if __name__ == "__main__":
    score_plot()
