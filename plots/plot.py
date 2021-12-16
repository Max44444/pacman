from matplotlib import pyplot as plt
import sys

def main():
    filepath = sys.argv[1]
    episodes = []
    window_losses = []
    avg_score = []
    window_avg_score = []

    with open(filepath, encoding='utf-16') as f:
        lines = f.read()
        lines = lines.split("\n")[:-1]
        for line in lines:
            data = line.split()
            episodes.append(int(data[0]))
            window_losses.append(int(data[1]))
            avg_score.append(float(data[2]))
            window_avg_score.append(float(data[3]))

    draw(episodes, window_losses, "No. of Episodes", "Total losses in last 10 games", "Total losses in last 10 games")
    draw(episodes, avg_score, "No. of Episodes", "Running Average Score", "Running Average Score")
    draw(episodes, window_avg_score, "No. of Episodes", "Average Score in last 10 Episodes", "Average Score in last 10 Episodes")


def draw(x, y, x_label, y_label, title):
    plt.plot(x, y, label='Normal', linewidth=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()
    plt.gcf().clear()


if __name__ == '__main__':
    main()