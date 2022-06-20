
from torch.functional import tensordot
from Experiments.GridWorldExperiment import RunExperiment as GridWorld_RunExperiment
from Experiments.TwoWayGridExperiment import RunExperiment as TwoWayGrid_RunExperiment
from Experiments.MiniAtariExperiment import RunExperiment as MiniAtari_RunExperiment

from Agents.ImperfectDQNMCTSAgentMiniAtari import *




if __name__ == '__main__':

    experiment = MiniAtari_RunExperiment()
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
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Depth20_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth20_ParameterStudy_run1.p',

        # 'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=2_S=N_B=N_TrueUncertainty_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_ParameterStudy_run1.p',
        'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_ParameterStudy_run1.p',
    ]

    results_UAMCTS_Freeway_Offline=[
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',

        # 'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=2_S=N_B=N_TrueUncertainty_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_ParameterStudy_run1.p',
        'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_ParameterStudy_run1.p',
    ]
    results_UAMCTS_Breakout_Offline=[
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Depth50_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_TrueModel_Depth50_ParameterStudy_run1.p',
        
        # 'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=2_S=N_B=N_TrueUncertainty_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=N_S=N_B=N_TrueUncertainty_ParameterStudy_run1.p',
        'Breakout_CorruptedStates=[2, 4]_MCTS_R=5_E=2_S=1_B=1_TrueUncertainty_ParameterStudy_run1.p',
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

    # exp_names = ['C', 'T', 'Backpropagation', 'Selection', 'Expansion', 'Simulation', 'Combined']
    # exp_names = ['C', 'T', 'Combined Online']
    # exp_names = ['Backpropagation', 'Selection', 'Expansion', 'Simulation', 'Combined']

    exp_names_DQMCTS_Space = [
                'dqn3', 'dqn7', 'dqn20',
                'C', 'T',
                's3E1', 's7E1', 's20E1',
                'S3C', 'S7C', 'S20C']
    
    exp_names_DQMCTS_Freeway = [
                'dqn7', 'dqn10', 'dqn20',
                'C', 'T',
                'F7E1', 'F10E1', 'F20E1',
                'F7C', 'F10C', 'F20C']
    
    exp_names_DQMCTS_Breakout = [
                'dqn7', 'dqn10', 'dqn20',
                'C', 'T',
                'B7E1', 'B10E1', 'B20E1',
                'B7C', 'B10C', 'B20C']

    exp_names_MCTS_Space = [
        'C_0', 'T_0', 'C_5', 'T_5', 'C_10', 'T_10', 'C_20', 'T_20', 'C_50', 'T_50'
    ]
    exp_names_MCTS_Freeway = [
        'C_0', 'T_0', 'C_5', 'T_5', 'C_10', 'T_10', 'C_25', 'T_25', 'C_50', 'T_50'
    ]
    exp_names_MCTS_Breakout = [
        'C_0', 'T_0', 'C_5', 'T_5', 'C_10', 'T_10', 'C_25', 'T_25', 'C_50', 'T_50'
    ]

    exp_names_UAMCTS_Offline = [
        'C', 'T', 
        'E',
        'R', 'UAMCTS'
    ]
    exp_names_UAMCTS_online = [
        'C', 'T',
        'e1000', 'r1000', 'u1000', 
        'e3000', 'r3000', 'u3000', 
        'e7000', 'r7000', 'u7000',
    ]
    # experiment.show_multiple_experiment_result_paper(results_UAMCTS_Breakout_Offline, exp_names_UAMCTS_Offline)

    experiment.show_multiple_experiment_result_paper(results_UAMCTS_Freeway_Online, exp_names_UAMCTS_online)
    # experiment.multiple_experiments_t_test(results_file_name_list, exp_names)

