import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import random


def plot_result(file_name, agent_name, is_offline, metric, axs_test, index=0):
    results_dir = "CombinedResults/"

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15) 

    with open(results_dir + file_name, 'rb') as f:
        result = pickle.load(f)
        print(file_name, result[metric].shape)
        result = result[metric][index]

    def make_smooth(runs, s=5):
        smooth_runs = np.zeros([runs.shape[0], runs.shape[1] - s])
        for i in range(runs.shape[0]):
            for j in range(runs.shape[1] - s):
                smooth_runs[i, j] = np.mean(runs[i, j: j + s])
        return smooth_runs

    def offline(metrics, agent_name, axs_test):
        print(metrics.shape)
        # metrics = np.array([np.mean(metrics, axis=1)])
        print(metrics.shape)
        # metrics_avg = np.mean(metrics, axis=1)
        # metrics_std = np.std(metrics, axis=1)        
        metrics_avg = np.mean(metrics)
        metrics_std = np.std(metrics)
        number_of_colors = 8

        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(number_of_colors)])
        axs_test.scatter(y=metrics_avg, x=agent_name, color=color)
        axs_test.axhline(metrics_avg, label=agent_name, color=color, linestyle="--")
        axs_test.errorbar(y=metrics_avg, x=agent_name, yerr=metrics_std,
                            ls="-", color=color, capsize=5)
    
    def online(metrics, agent_name, axs_test):
        metrics = make_smooth(metrics, s=50)

        metrics_avg = np.mean(metrics, axis=0)
        metrics_std = np.std(metrics, axis=0)
        x = range(len(metrics_avg))
        number_of_colors = 8

        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(number_of_colors)])
             
        axs_test.plot(x, metrics_avg, label=agent_name, color=color)
        axs_test.fill_between(x,
                        metrics_avg - metrics_std,
                        metrics_avg + metrics_std, color=color,
                        alpha=.3, edgecolor='none')

    if is_offline:
        offline(result, agent_name, axs_test)
    else:
        online(result, agent_name, axs_test)


if __name__ == '__main__':
    # todo: add smoothness
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, required=True)
    # parser.add_argument('--file_name', type=str, required=True)
    # parser.add_argument('--plot_name', type=str, required=True)
    # parser.add_argument('--metric', type=str, required=True)
    # args = parser.parse_args()

    # is_offline = args.scenario == "offline"
    # print(args.metric)
    # plot_result(args.file_name, args.plot_name, is_offline, args.metric)


    plots_dir = "Plots/"
    plot_name = "twowayicy_test"   
    fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)
    plot_result("MCTS_TwoWayIcy_TrueModel.p", "T", True, "num_steps", axs_test)
    plot_result("MCTS_TwoWayIcy.p", "C", True, "num_steps", axs_test)
    plot_result("UAMCTS_TwoWayIcy.p", "UAMCTS", False, "num_steps", axs_test)
    axs_test.legend()
    fig_test.savefig(plots_dir + plot_name + ".png", format="png")


    # plots_dir = "Plots/"
    # plot_name = "twowayicyv2"   
    # fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)
    # plot_result("MCTS_TwoWayIcyV2_TrueModel.p", ["T"], True, "num_steps", axs_test)
    # plot_result("MCTS_TwoWayIcyV2.p", ["C"], True, "num_steps", axs_test)
    # plot_result("UAMCTS_TwoWayIcyV2_Offline_Run0.p", ["UAMCTS"], True, "num_steps", axs_test)
    # axs_test.legend()
    # fig_test.savefig(plots_dir + plot_name + ".png", format="png")
    # fig_test.savefig(plots_dir + plot_name + ".svg", format="svg")



    # plots_dir = "Plots/"
    # plot_name = "test"   
    # fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)
    # for index in range(4):
    #     plot_result("MCTS_TwoWayIcyV2_ParamStudy_Run0.p", str(index), True, "num_steps", axs_test, index=index)

    # axs_test.legend()
    # fig_test.savefig(plots_dir + plot_name + ".png", format="png")