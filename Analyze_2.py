import matplotlib.pyplot as plt
import argparse
import pickle

from Experiments.TwoWayGridExperiment import RunExperiment as TwoWayGrid_RunExperiment
from Experiments.MinAtarExperiment import RunExperiment as MinAtar_RunExperiment


def combine_experiment_result(runs_list, result_file_name):
    res = {'num_steps': None, 'rewards': None, 'experiment_objs': None, 'detail': None}
    for file_name in runs_list:
        print('hello')
        with open("FinalResults/" + file_name, 'rb') as f:
            result = pickle.load(f)
        f.close()
        if res['num_steps'] is None:
            res['num_steps'] = result['num_steps']
        else:
            res['num_steps'] = np.concatenate([res['num_steps'], result['num_steps']], axis=1)
        if res['rewards'] is None:
            res['rewards'] = result['rewards']
        else:
            res['rewards'] = np.concatenate([res['rewards'], result['rewards']], axis=1)
        if res['experiment_objs'] is None:
            res['experiment_objs'] = result['experiment_objs']
    with open("PlotResults/" + result_file_name + '.p', 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--plot_name', type=str, required=True)

    experiment = MinAtar_RunExperiment()
    # experiment = TwoWayGrid_RunExperiment()

    results_UAMCTS_Space_Offline=[
        'SpaceInvaders_MCTS_TrueModel_Combined',
        'SpaceInvaders_MCTS_CorruptedModel_Combined',
        'SpaceInvaders_UAMCTS_Selection_Run0',
        'SpaceInvaders_UAMCTS_Expansion_Run0',
        'SpaceInvaders_UAMCTS_Simulation_Run0',
        'SpaceInvaders_UAMCTS_Backpropagation_Run0',
        'SpaceInvaders_UAMCTS_Combined_Run0']
    results_UAMCTS_Space_Online = [ 
        '2_SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Combined'
    ]
    results_UAMCTS_Freeway_Offline=[
        'Freeway_MCTS_TrueModel_Combined',
        'Freeway_MCTS_CorruptedModel_Combined',
        'Freeway_UAMCTS_Selection_Run0',
        'Freeway_UAMCTS_Expansion_Run0',
        'Freeway_UAMCTS_Simulation_Run0',
        'Freeway_UAMCTS_Backpropagation_Run0',
        'Freeway_UAMCTS_Combined_Run0',
    ]
    results_UAMCTS_Freeway_Online = [ 
        '2_Freeway_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Combined'
    ]    
    results_UAMCTS_Breakout_Offline=[
        '8_Breakout_MCTS_TrueModel_Combined',
        '8_Breakout_MCTS_CorruptedModel_Combined',
        '8_Breakout_UAMCTS_Selection_Combined',
        '8_Breakout_UAMCTS_Expansion_Combined',
        '8_Breakout_UAMCTS_Simulation_Combined',
        '8_Breakout_UAMCTS_Backpropagation_Combined',
        '8_Breakout_UAMCTS_Combined_Combined',
    ]
    results_UAMCTS_Breakout_Online = [ 
        '3_Breakout_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Combined'
    ]   

    exp_names = ['T', 'C', 'S', 'E', 'R', 'B', 'U']
    
    fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plot_name = "Breakout_F"
    axs_test.xaxis.tick_top() 
    experiment.show_multiple_experiment_result_paper(results_UAMCTS_Breakout_Offline, exp_names, plot_name, fig_test, axs_test, is_offline=True)
    # exit(0)
    axs_test = axs_test.twiny()
    axs_test.xaxis.tick_bottom() 

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    experiment.show_multiple_experiment_result_paper(results_UAMCTS_Breakout_Online, ["Online"], plot_name, fig_test, axs_test, is_offline=False)


    fig_test.savefig("Plots/" + plot_name +".png", format="png")
    fig_test.savefig("Plots/" + plot_name +".svg", format="svg")
    exit(0)
    
    # experiment.show_multiple_experiment_result_paper(results_UAMCTS_Space_Offline, exp_names, "SpaceInvaders-Offline")
    # experiment.show_multiple_experiment_result_paper(results_UAMCTS_Breakout_Offline, exp_names, "Breakout-Offline")
    # exit(0)
    # combining_results = [
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_5000_64_Run0.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_5000_64_Run1.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_5000_64_Run2.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_5000_64_Run3.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_5000_64_Run4.p"
    # ]

    # combining_results = [
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_10000_64_Run0.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_10000_64_Run1.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_10000_64_Run2.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_10000_64_Run3.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_10000_64_Run4.p"
    # ]

    # combining_results = [
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_25000_64_Run0.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_25000_64_Run1.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_25000_64_Run2.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_25000_64_Run3.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_2500_25000_64_Run4.p"
    # ]

    # combining_results = [
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_5000_10000_64_Run0.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_5000_10000_64_Run1.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_5000_10000_64_Run2.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_5000_10000_64_Run3.p",
    #     "Freeway_SemiOnlineUAMCTS_R5_E=N_S=N_B=N_5000_10000_64_Run4.p"
    # ]

    # run_list = [1, 2, 4, 6, 7, 8, 9]
    # combining_results = ['V2-Freeway_SemiOnlineUAMCTS_R=N_E=2_S=N_B=N_AdaptiveTau=10_5000_5000_Run' + str(i) + '.p' for i in run_list]
    run_list = [0, 1, 2, 3, 4]

    # combining_results = ['SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run' + str(i) + '.p' for i in run_list]
    # run_list = [0:10]
    combining_results = ['3_Breakout_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run' + str(i) + '.p' for i in list(range(8)) + list(range(50,55))  ]
    combined_name = "3_Breakout_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Combined"    
    # combining_results = ['Freeway_Corrupted_' + str(i) + '.p' for i in range(5)]
    # combined_name = "2_Freeway_CorruptedMCTS"    
    # combining_results = ['Breakout_MCTS_CorruptedModel_Run' + str(i) + '.p' for i in range(1)]
    # combined_name = "Breakout_MCTS_CorruptedModel_Combined"
    combine_runs(combining_results, combined_name)
    # exit(0)

    # experiment.show_multiple_experiment_result_paper([combined_name], ['UAMCTS'], "SpaceInvaders-SemiOnline")
    # experiment.show_multiple_experiment_result_paper([combined_name], ['UAMCTS'], "Freeway-SemiOnline")
    # experiment.show_multiple_experiment_result_paper([combined_name], ['UAMCTS'], "Breakout-SemiOnline")



# UAMCTS/Results/Freeway_SemiOnlineUAMCTS_R=N_E=2_S=N_B=N_OneTime_Tau=1_1000_5000_32_Run_Combined.p