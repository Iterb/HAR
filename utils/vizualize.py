
from matplotlib import pyplot as plt
import numpy as np
def plot_probablites(data):
    #for y in 
    x = np.arange(0.5, len(data)+0.5, 1)
    plt.plot(x ,data)
    plt.legend(['punch', 'kicking', 'pushing', 'pat on back', 'point finger',
            'hugging', 'giving an object', 'touch pocket',
            'shaking hands', 'walking towards', 'walking apart'])
    plt.savefig("test.png")