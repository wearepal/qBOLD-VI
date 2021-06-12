import argparse

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualise a prediction\'s certainty via normal pdf')

    parser.add_argument('-i',
                        required=True,
                        help='index of the prediction in files')

    args = parser.parse_args()
    idx = int(args.i)

    with open('params.csv') as fp:
        for i, line in enumerate(fp):
            if i == idx:
                sample = float(line.split(',')[0])
                break

    with open('predictions.csv') as fp:
        for i, line in enumerate(fp):
            if i == idx:
                mu = float(line.split(',')[0])
                sigma = np.exp(float(line.split(',')[1]))
                break

    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    print(sigma)
    for i in range(-3, 4):
        plt.axvline(mu+i*sigma, color='red', ls='--')
    plt.axvline(sample, color='green')
    plt.savefig('certainty_plot.jpg')
