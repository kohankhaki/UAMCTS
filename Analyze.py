import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np


def plot_result(file_name, plot_name, is_offline, metric):
    plots_dir = "Plots/"
    results_dir = "CombinedResults/"   
    fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15) 

    with open(results_dir + file_name, 'rb') as f:
        result = pickle.load(f)
        result = result[metric]

    def make_smooth(runs, s=5):
        smooth_runs = np.zeros([runs.shape[0], runs.shape[1] - s])
        for i in range(runs.shape[0]):
            for j in range(runs.shape[1] - s):
                smooth_runs[i, j] = np.mean(runs[i, j: j + s])
        return smooth_runs

    def offline(metrics, plot_name, fig_test, axs_test):
        metrics = np.array([np.mean(metrics, axis=0)])

        metrics_avg = np.mean(metrics[0], axis=0)
        metrics_std = np.std(metrics[0], axis=0)
        axs_test.scatter(y=metrics_avg, x="Agent", color="#e0030c")
        axs_test.axhline(metrics_avg, label="Agent", color="#e0030c", linestyle="--")
        axs_test.errorbar(y=metrics_avg, x="Agent", yerr=metrics_std,
                            ls='none', color="#e0030c", capsize=5)

        axs_test.legend()
        fig_test.savefig(plots_dir + plot_name +".png", format="png")
    
    def online(metrics, plot_name, fig_test, axs_test):
        print(metrics.shape)
        metrics = np.array([make_smooth(metrics[0], s=50)])
        print(metrics.shape)

        metrics_avg = np.mean(metrics[0], axis=0)
        metrics_std = np.std(metrics[0], axis=0)
        x = range(len(metrics_avg))
        axs_test.plot(x, metrics_avg, label="Agent", color="orchid")
        axs_test.fill_between(x,
                        metrics_avg - metrics_std,
                        metrics_avg + metrics_std, color="orchid",
                        alpha=.3, edgecolor='none')
        axs_test.legend()
        fig_test.savefig(plots_dir + plot_name +".png", format="png")

    if is_offline:
        offline(result, plot_name, fig_test, axs_test)
    else:
        online(result, plot_name, fig_test, axs_test)


if __name__ == '__main__':
    # todo: add smoothness
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--plot_name', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    args = parser.parse_args()

    is_offline = args.scenario == "offline"
    print(args.metric)
    plot_result(args.file_name, args.plot_name, is_offline, args.metric)