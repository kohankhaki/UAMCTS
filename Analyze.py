
from torch.functional import tensordot
# from Experiments.GridWorldExperiment import RunExperiment as GridWorld_RunExperiment
# from Experiments.TwoWayGridExperiment import RunExperiment as TwoWayGrid_RunExperiment
from Experiments.MinAtarExperiment import RunExperiment as MinAtar_RunExperiment

from Agents.ImperfectDQNMCTSAgentMinAtar import *

def combine_runs(runs_list, result_name):
    num_runs = len(runs_list)
    combined_runs = []
    num_episode = 0
    for run in runs_list:
        with open("Results/" + run, 'rb') as f:
            run_result = pickle.load(f)
            for i in range(run_result['rewards'][0].shape[0]):
                combined_runs.append(run_result['rewards'][0][i])
            num_episode = run_result['rewards'][0].shape[1]

    combined_runs = np.array(combined_runs)
    combined_runs = np.expand_dims(combined_runs, axis=0) 
    
    with open("Results/" + result_name + '.p', 'wb') as f:
        result = run_result
        result['rewards'] = combined_runs
        pickle.dump(result, f)
            



if __name__ == '__main__':

    experiment = MinAtar_RunExperiment()
    # experiment = TwoWayGrid_RunExperiment()


    results_file_name_list = [
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_BestParameter',
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_ParameterStudy',

        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_run1',
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_run1',
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=2_S=N_B=N_TrueUncertainty_run1',
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_run1',
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty',
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1',
        
        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_Online_PretrainedUncertaintyEp20'
       
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_BestParameter',
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_ParameterStudy',

        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_run1',
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_run1',
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=2_S=N_B=N_TrueUncertainty_run1',
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_run1',
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_run1',
        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1',

        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_Online_PretrainedUncertaintyEp90'

        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_BestParameter',
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_ParameterStudy',

        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_run1',
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_run1',
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=2_S=N_B=N_TrueUncertainty_run1',
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_run1',
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty',
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1',

        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_Online_PretrainedUncertaintyEp90'

    ]

    results_DQMCTS_Space_D0 = [
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E3000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth0_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth0_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth0_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth0_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth0_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth0_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth0_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth0_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Space_D5 = [
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E3000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth5_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth5_ParameterStudy_run1.p',
        
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth5_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth5_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth5_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth5_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth5_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth5_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Space_D10 = [
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E3000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',
       
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth10_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth10_ParameterStudy_run1.p',
    
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth10_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth10_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth10_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth10_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth10_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth10_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Space_D20 = [
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E3000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',
       
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth20_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth20_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN3k_64x64_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth20_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Freeway_D0 = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth0_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth0_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth0_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth0_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth0_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth0_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth0_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth0_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Freeway_D5 = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth5_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth5_ParameterStudy_run1.p',
  
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth5_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth5_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth5_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth5_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth5_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth5_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Freeway_D10 = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth10_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth10_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth10_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth10_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth10_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth10_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth10_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth10_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Freeway_D25 = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth25_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth25_ParameterStudy_run1.p',
      
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth25_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth25_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth25_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth25_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth25_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth25_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Freeway_D50 = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth50_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth50_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Breakout_D0 = [
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth0_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth0_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth0_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth0_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth0_ParameterStudy_run1.p',
        
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth0_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth0_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth0_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Breakout_D5 = [
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth5_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth5_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth5_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth5_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth5_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth5_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth5_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth5_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Breakout_D10 = [
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth10_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth10_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth10_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth10_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth10_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth10_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth10_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth10_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Breakout_D25 = [
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth25_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth25_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth25_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth25_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth25_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth25_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth25_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth25_ParameterStudy_run1.p',
    ]
    results_DQMCTS_Breakout_D50 = [
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E7000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E10000_64x64_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1.p',
        
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth50_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN7k_64x64_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN10k_64x64_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=4_E=1_S=N_B=N_PreTrainedDQN20k_64x64_Depth50_ParameterStudy_run1.p',
    ]
    results_MCTS_Space = [
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth0_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth0_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth5_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth5_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth10_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth10_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth20_ParameterStudy_run1.p',

        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',
    ]
    results_MCTS_Freeway = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth0_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth0_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth5_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth5_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth10_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth10_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth25_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth25_ParameterStudy_run1.p',

        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',
    ]
    results_MCTS_Breakout = [
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth0_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth0_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth5_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth5_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth10_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth10_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth25_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth25_ParameterStudy_run1.p',

        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',
    ]
    results_UAMCTS_Space_Offline=[
        '1_SpaceInvaders_TrueMCTS',
        '1_SpaceInvaders_CorruptedMCTS',
        '1_SpaceInvaders_OfflineUAMCTS_R=N_E=N_S=1_B=N',
        '1_SpaceInvaders_OfflineUAMCTS_R=N_E=2_S=N_B=N',
        '1_SpaceInvaders_OfflineUAMCTS_R=5_E=N_S=N_B=N',
        '1_SpaceInvaders_OfflineUAMCTS_R=N_E=N_S=N_B=1',
        '1_SpaceInvaders_OfflineUAMCTS_R=5_E=2_S=1_B=1',
    ]
    results_UAMCTS_Freeway_Offline=[
        '1_Freeway_TrueMCTS',
        '2_Freeway_CorruptedMCTS',
        '1_Freeway_OfflineUAMCTS_R=N_E=N_S=1_B=N',
        '1_Freeway_OfflineUAMCTS_R=N_E=2_S=N_B=N',
        '1_Freeway_OfflineUAMCTS_R=5_E=N_S=N_B=N',
        '1_Freeway_OfflineUAMCTS_R=N_E=N_S=N_B=1',
        '1_Freeway_OfflineUAMCTS_R=5_E=2_S=1_B=1',
    ]
    results_UAMCTS_Breakout_Offline=[
        '1_Breakout_TrueMCTS',
        '1_Breakout_CorruptedMCTS',
        '1_Breakout_OfflineUAMCTS_R=N_E=N_S=1_B=N',
        '1_Breakout_OfflineUAMCTS_R=N_E=2_S=N_B=N',
        '1_Breakout_OfflineUAMCTS_R=5_E=N_S=N_B=N',
        '1_Breakout_OfflineUAMCTS_R=N_E=N_S=N_B=1',
        '1_Breakout_OfflineUAMCTS_R=5_E=2_S=1_B=1',
    ]
    results_UAMCTS_Space_Online = [
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth20_ParameterStudy_run1.p',

        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B1000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B1000_ParameterStudy',

        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B3000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B3000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B3000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B3000_ParameterStudy',

        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B7000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B7000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B7000_ParameterStudy',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B7000_ParameterStudy',

    ]
    results_UAMCTS_Freeway_Online = [
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',


        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B1000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B1000_ParameterStudy',


        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B3000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B3000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B3000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B3000_ParameterStudy',

        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B7000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B7000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B7000_ParameterStudy',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B7000_ParameterStudy',

    ]
    results_UAMCTS_Breakout_Online = [
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',


        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B1000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B1000_ParameterStudy',


        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B3000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B3000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B3000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B3000_ParameterStudy',

        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B7000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B7000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B7000_ParameterStudy',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrainedUncertainty_B7000_ParameterStudy',




    ]

    exp_names = ['T', 'C', 'Selection', 'Expansion', 'Simulation', 'Backpropagation', 'Combined']
    experiment.show_multiple_experiment_result_paper(results_UAMCTS_Breakout_Offline, exp_names, "ttt")
    exit(0)
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
    # combining_results = ['2_SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run' + str(i) + '.p' for i in range(6)]
    # combined_name = "2_SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Combined"    
    # combining_results = ['Freeway_Corrupted_' + str(i) + '.p' for i in range(5)]
    # combined_name = "2_Freeway_CorruptedMCTS"    
    combining_results = ['2_Freeway_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run' + str(i) + '.p' for i in range()]
    combined_name = "2_Breakout_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Combined"
    combine_runs(combining_results, combined_name)
    # experiment.show_multiple_experiment_result_paper([combined_name], ['UAMCTS'], "SpaceInvaders-SemiOnline")
    # experiment.show_multiple_experiment_result_paper([combined_name], ['UAMCTS'], "Freeway-SemiOnline")
    experiment.show_multiple_experiment_result_paper([combined_name], ['UAMCTS'], "Breakout-SemiOnline")



# UAMCTS/Results/Freeway_SemiOnlineUAMCTS_R=N_E=2_S=N_B=N_OneTime_Tau=1_1000_5000_32_Run_Combined.p