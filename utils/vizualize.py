from matplotlib import pyplot as plt
import numpy as np


def plot_probablites(data):
    # for y in
    x = np.arange(0.5, len(data) + 0.5, 1)
    plt.figure(figsize=(12, 8))
    plt.plot(x, data)
    ax = plt.subplot(111)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.legend( 
        [
            "punch",
            "kicking",
            "pushing",
            "pat on back",
            "point finger",
            "hugging",
            "giving an object",
            "touch pocket",
            "shaking hands",
            "walking towards",
            "walking apart",
        ],
        bbox_to_anchor=(1.05, 0.6))
    plt.ylabel('Uśrednione prawdopodobieństwo')
    plt.xlabel('Czas [s]')
    plt.savefig("test.png")
