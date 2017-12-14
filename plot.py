import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine_path = "data/winequality-white.csv"

num_kfolds = 10


def make_plots(scores):
    for key in scores:
        scr = scores[key]
        x_vals = scr.keys()
        y_vals = scr.values()
        plt.bar(x_vals, y_vals)
        plt.xlabel("Margin of Error")
        plt.ylabel("Percent Coverage")


def main():
    dataset = pd.read_csv(wine_path, delimiter=';')
    all_headers = list(dataset.columns.values)

    for headers in np.array_split(all_headers, 3):
        f, a = plt.subplots(2, 2)
        a = a.ravel()

        for idx, ax in enumerate(a):
            header = headers[idx]
            ax.hist(dataset[header], alpha=0.45)
            ax.set_title(header)
        plt.tight_layout()
        plt.show()

    for headers in np.array_split(all_headers, 3):
        f, a = plt.subplots(2, 2)
        a = a.ravel()

        for idx, ax in enumerate(a):
            header = headers[idx]
            ax.boxplot(dataset[header], 0)
            ax.set_title(header)
        plt.show()


if __name__ == '__main__':
    main()
