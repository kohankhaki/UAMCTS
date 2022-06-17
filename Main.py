'''
train a DQN with MCTS
See if DQN agrees with DQN (saving the best path)
At what step DQN starts to work with MCTS
'''
# use both mcts trajectories and dqn trajectories in the dqn buffer
# use the selection path but with all children in the path
# rollout with dqn policy in mcts
import threading
import time

import utils, config

from Experiments.ExperimentObject import ExperimentObject
from Experiments.GridWorldExperiment import RunExperiment as GridWorld_RunExperiment
from Experiments.TwoWayGridExperiment import RunExperiment as TwoWayGrid_RunExperiment
from Experiments.MinAtarExperiment import RunExperiment as MinAtar_RunExperiment

from Agents.ImperfectDQNMCTSAgentMinAtar import *
from Agents.SemiOnlineUAMCTS import *
from Agents.DynaAgent import *

if __name__ == '__main__':

    agent_class_list = [SemiOnlineUAMCTS]
    # agent_class_list = [RealBaseDynaAgent]

    s_vf_list = config.s_vf_list
    s_md_list = config.s_md_list
    model_corruption_list = config.model_corruption_list
    experiment_detail = config.experiment_detail

    c_list = config.c_list
    
    num_iteration_list = config.num_iteration_list
    simulation_depth_list = config.simulation_depth_list
    num_simulation_list = config.num_simulation_list
    tau_list = config.tau_list

    
    model_list = config.model_list


    vf_list = config.trained_vf_list
    experiment = MinAtar_RunExperiment()

    experiment_object_list = []
    for agent_class in agent_class_list:
        for s_vf in s_vf_list:
            for model in model_list:
                for vf in vf_list:
                    for s_md in s_md_list:
                        for c in c_list:
                            for num_iteration in num_iteration_list:
                                for simulation_depth in simulation_depth_list:
                                    for num_simulation in num_simulation_list:
                                        for model_corruption in model_corruption_list:
                                            for tau in tau_list:
                                                params = {'pre_trained': None,
                                                        'vf_step_size': s_vf,
                                                        'vf': vf,
                                                        'model': model,
                                                        'model_step_size': s_md,
                                                        'c': c,
                                                        'num_iteration': num_iteration,
                                                        'simulation_depth': simulation_depth,
                                                        'num_simulation': num_simulation,
                                                        'model_corruption': model_corruption,
                                                        'tau': tau,}
                                                obj = ExperimentObject(agent_class, params)
                                                experiment_object_list.append(obj)

    result_file_name = config.result_file_name

    experiment.run_experiment(experiment_object_list, result_file_name=result_file_name, detail=experiment_detail)
