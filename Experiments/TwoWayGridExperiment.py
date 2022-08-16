from Environments.GridWorldBase import GridWorld
import numpy as np
import torch
import os
import utils, config
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter

from Experiments.BaseExperiment import BaseExperiment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
debug = False


class TwoWayGridExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)

        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.num_samples = 0
        self.device = device
        self.visited_states = np.array([[0, 0, 0, 0]])

    def start(self):
        self.num_steps = 0
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action)
        self.num_samples += 1
        obs = self.observationChannel(s)
        self.total_reward += reward

        if self._render_on and self.num_episodes >= 0:
            self.environment.render()

        if term:
            self.agent.end(reward)
            roat = (reward, obs, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, obs)
            roat = (reward, obs, self.last_action, term)

        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def runEpisode(self, max_steps=0):
        is_terminal = False
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        self.num_episodes += 1
        self.num_steps_to_goal_list.append(self.num_steps)
        if debug:
            print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return np.asarray(s)

    def recordTrajectory(self, s, a, r, t):
        pass


class RunExperiment():
    def __init__(self):
        self.device = torch.device("cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)

    def run_experiment(self, experiment_object_list, result_file_name, detail=None):
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        for i, obj in tqdm(enumerate(experiment_object_list)):
            print("---------------------")
            print("This is the case: ", i)

            for r in range(num_runs):
                print("starting runtime ", r + 1)
                random_obstacle_x = 3#np.random.randint(0, 8)
                random_obstacle_y = 0#np.random.choice([0, 2])
                env = GridWorld(params={'size': (3, 8), 'init_state': (1, 0), 'state_mode': 'coord',
                                      'obstacles_pos': [(1, 1),(1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                                                        (random_obstacle_y, random_obstacle_x)],
                                      'rewards_pos': [(1, 7)], 'rewards_value': [10],
                                      'terminals_pos': [(1, 7)], 'termination_probs': [1],
                                      'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                                      'neighbour_distance': 0,
                                      'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1],
                                      'transition_randomness': 0.0,
                                      'window_size': (255, 255),
                                      'aging_reward': -1
                                      })

                corrupt_env = GridWorld(params={'size': (3, 8), 'init_state': (1, 0), 'state_mode': 'coord',
                                        'obstacles_pos': [(1, 1),(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)],
                                        'rewards_pos': [(1, 7)], 'rewards_value': [10],
                                        'terminals_pos': [(1, 7)], 'termination_probs': [1],
                                        'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                                        'neighbour_distance': 0,
                                        'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0],
                                        'obstacle_color': [1, 1, 1],
                                        'transition_randomness': 0.0,
                                        'window_size': (255, 255),
                                        'aging_reward': -1
                                        })
                # initializing the agent
                agent = obj.agent_class({'action_list': env.getAllActions(),
                                         'gamma': 1.0, 'epsilon_max': 0.9, 'epsilon_min': 0.1, 'epsilon_decay': 200,
                                         'tau': obj.tau,
                                         'model_corruption': obj.model_corruption,
                                         'max_stepsize': obj.vf_step_size,
                                         'model_stepsize': obj.model_step_size,
                                         'reward_function': None,
                                         'goal': None,
                                         'device': self.device,
                                         'model': obj.model,
                                         'true_bw_model': None,
                                         'true_fw_model': env.fullTransitionFunction,
                                         'corrupted_fw_model':corrupt_env.fullTransitionFunction,
                                         'transition_dynamics': None,
                                         'c': obj.c,
                                         'num_iteration': obj.num_iteration,
                                         'simulation_depth': obj.simulation_depth,
                                         'num_simulation': obj.num_simulation,
                                         'vf': obj.vf_network, 'dataset': None})

                # initialize experiment
                experiment = TwoWayGridExperiment(agent, env, self.device)
                for e in range(num_episode):
                    if debug:
                        print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps

                    # with torch.no_grad():
                    #     # print value function
                    #     for s in env.getAllStates():
                    #         s_rep = agent.getStateRepresentation(s)
                    #         ensemble_values = np.asarray([agent._vf['q']['network'][i](s_rep).numpy() for i in
                    #                            range(agent._vf['q']['num_ensembles'])])
                    #         avg_values = np.mean(ensemble_values, axis=0)
                    #         std_values = np.std(ensemble_values, axis=0)
                    #         s_value = np.mean(avg_values)
                    #         s_uncertainty = np.mean(std_values)
                            # print(s_rep, s_value, s_uncertainty)
                            # for a in env.getAllActions():
                            #     a_index = env.getActionIndex(a)
                            #     q_value = value[0][a_index].item()
                            #     print(s_rep, a_index, q_value)

                # if True:
                #     agent.saveValueFunction("TwoWayGridResult/DQN_VF_test")
                #     for s in env.getAllStates():
                #         s_rep = agent.getStateRepresentation(s)
                #         values = agent._vf['q']['network'](s_rep)
                #         print(s_rep, "---",values)
        with open("TwoWayGridResult/" + result_file_name + '.p', 'wb') as f:
            result = {'num_steps': self.num_steps_run_list,
                      'experiment_objs': experiment_object_list,
                      'detail': detail,}
            pickle.dump(result, f)
        f.close()
        # show_num_steps_plot(self.num_steps_run_list, ["Uncertain_MCTS"])
        # save_num_steps_plot(self.num_steps_run_list, experiment_object_list)

    def show_experiment_result(self, result_file_name):
        with open("TwoWayGridResult/" + result_file_name + '.p', 'rb') as f:
            result = pickle.load(f)
        f.close()
        # show_num_steps_plot(result['num_steps'], result['experiment_objs'])
        save_num_steps_plot(result['num_steps'], result['experiment_objs'], result_file_name)

    def show_multiple_experiment_result(self, results_file_name_list, exp_names):
        def find_best_c(num_steps, experiment_objs):
            removed_list = []
            print(num_steps.shape)
            num_steps_avg = np.mean(num_steps, axis=1)
            for counter1, i in enumerate(experiment_objs):
                for counter2, j in enumerate(experiment_objs):
                    if i.num_iteration == j.num_iteration and \
                    i.num_simulation == j.num_simulation and \
                    i.simulation_depth == j.simulation_depth and \
                    i.tau == j.tau and \
                    i.c != j.c:
                        if num_steps_avg[counter1] > num_steps_avg[counter2]:
                            removed_list.append(counter1)
                        else:
                            removed_list.append(counter2)
            num_steps = np.delete(num_steps, removed_list, 0)
            experiment_objs_new = []
            for i, obj in enumerate(experiment_objs):
                if i not in removed_list:
                    experiment_objs_new.append(obj)
            return num_steps, experiment_objs_new

        def find_best_tau(num_steps, experiment_objs):
            removed_list = []
            num_steps_avg = np.mean(num_steps, axis=1)
            for counter1, i in enumerate(experiment_objs):
                for counter2, j in enumerate(experiment_objs):
                    if i.num_iteration == j.num_iteration and \
                    i.num_simulation == j.num_simulation and \
                    i.simulation_depth == j.simulation_depth and \
                    i.tau != j.tau:
                        if num_steps_avg[counter1] > num_steps_avg[counter2]:
                            removed_list.append(counter1)
                        else:
                            removed_list.append(counter2)
            num_steps = np.delete(num_steps, removed_list, 0)
            experiment_objs_new = []
            for i, obj in enumerate(experiment_objs):
                if i not in removed_list:
                    experiment_objs_new.append(obj)

            return num_steps, experiment_objs_new
        
        if len(results_file_name_list) != len(exp_names):
            print("experiments and names won't match", len(results_file_name_list), len(exp_names))
            return None
        
        fig_mean, axs_mean = plt.subplots(1, 1, constrained_layout=False)
        fig_std, axs_std = plt.subplots(1, 1, constrained_layout=False)
        fig_box, axs_box = plt.subplots(constrained_layout=False, figsize =(len(exp_names), 9))
        

        reward_list = []
        name_list = []
        axs_box.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
        axs_box.set(
            axisbelow=True,  # Hide the grid behind plot objects
            title='Comparison of different Algorithms',
            xlabel='Algorithm',
            ylabel='Reward',
            )
        for i in range(len(results_file_name_list)):
            result_file_name = results_file_name_list[i]
            exp_name = exp_names[i]
            result = self.combine_experiment_result(result_file_name)

            rewards, experiment_objs = result['num_steps'], result['experiment_objs']
            rewards, experiment_objs = find_best_c(rewards, experiment_objs)
            print(rewards.shape)
            rewards, experiment_objs = find_best_tau(rewards, experiment_objs)
            print(rewards.shape)
            names = experiment_obj_to_name(experiment_objs)
            
            for i, name in enumerate(names):
                rewards_avg = np.mean(rewards[i], axis=0)
                rewards_std = np.std(rewards[i], axis=0)
                x = range(len(rewards_avg))
                if len(rewards_avg) == 1:
                    color = generate_hex_color()
                    print(rewards_avg, rewards_std, exp_name+name, "\n")
                    if "True" in exp_name or "Corrupt" in exp_name:
                        reward_list.append(rewards[i, :, 0])
                        name_list.append(exp_name+name)
                        axs_mean.axhline(rewards_avg, label=exp_name+name, color=color, linestyle="--")
                        axs_std.axhline(rewards_std, label=exp_name+name, color=color, linestyle="--")
                    elif exp_name.count("=") == 1:
                        reward_list.append(rewards[i, :, 0])
                        name_list.append(exp_name+name)
                        axs_mean.axhline(rewards_avg, label=exp_name+name, color=color, linestyle=":")
                        axs_std.axhline(rewards_std, label=exp_name+name, color=color, linestyle=":")
                    elif exp_name.count("=") == 2:
                        reward_list.append(rewards[i, :, 0])
                        name_list.append(exp_name+name)
                        axs_mean.axhline(rewards_avg, label=exp_name+name, color=color, linestyle="-.")
                        axs_std.axhline(rewards_std, label=exp_name+name, color=color, linestyle="-.")
                    elif exp_name.count("=") == 3:
                        reward_list.append(rewards[i, :, 0])
                        name_list.append(exp_name+name)
                        axs_mean.axhline(rewards_avg, label=exp_name+name, color=color, linestyle=(0, (5, 5)))
                        axs_std.axhline(rewards_std, label=exp_name+name, color=color, linestyle=(0, (5, 5)))
                    elif exp_name.count("=") == 4:
                        reward_list.append(rewards[i, :, 0])
                        name_list.append(exp_name+name)
                        axs_mean.axhline(rewards_avg, label=exp_name+name, color=color, linestyle=(0, (10, 10)))
                        axs_std.axhline(rewards_std, label=exp_name+name, color=color, linestyle=(0, (10, 10)))
                    else:
                        reward_list.append(rewards[i, :, 0])
                        name_list.append(exp_name+name)
                        axs_mean.axhline(rewards_avg, label=exp_name+name, color=color, linestyle="-")
                        axs_std.axhline(rewards_std, label=exp_name+name, color=color, linestyle="-")
                    # axs_mean.axhspan(num_steps_avg - 0.1 * num_steps_std,
                    #             num_steps_avg + 0.1 * num_steps_std,
                    #             alpha=0.4, color=color)

                else:
                    axs_mean.plot(x, rewards_avg, label=exp_name+name)
                    axs_mean.fill_between(x,
                                    rewards_avg - 0.1 * rewards_std,
                                    rewards_avg + 0.1 * rewards_std,
                                    alpha=.4, edgecolor='none')
                axs_mean.legend()
                axs_std.legend()
        axs_box.set_xticklabels(name_list, rotation=45, fontsize=8)
        axs_box.boxplot(reward_list, showmeans=True)
        fig_mean.savefig("TwoWayGrid_average_performance"+".png")
        fig_std.savefig("TwoWayGrid_std_performance"+".png")
        fig_box.savefig("TwoWayGrid_boxplot"+".png")
    
    def combine_experiment_result(self, result_file_name):
        res = {'num_steps': None, 'rewards': None, 'experiment_objs': None, 'detail': None}
        all_files = os.listdir("TwoWayGridResult/")
        for file_name in all_files:
            if result_file_name in file_name:
                with open("TwoWayGridResult/" + file_name, 'rb') as f:
                    result = pickle.load(f)
                f.close()
                if res['num_steps'] is None:
                    res['num_steps'] = result['num_steps']
                else:
                    res['num_steps'] = np.concatenate([res['num_steps'], result['num_steps']], axis=1)
                
                if res['experiment_objs'] is None:
                    res['experiment_objs'] = result['experiment_objs']
        return res
def show_num_steps_plot(num_steps, agent_names):

    num_steps_avg = np.mean(num_steps, axis=1)
    writer = SummaryWriter()  # (comment="TuneModelwithDQN_LR{2**-4to-16}_BATCH{16}")
    counter = 0

    for agent in range(num_steps_avg.shape[0]):
        name = agent_names[agent]
        for i in range(num_steps_avg.shape[1]):
            writer.add_scalar(name, num_steps_avg[agent, i], counter)
            counter += 1
    writer.close()


def save_num_steps_plot(num_steps, experiment_objs, saved_name="test"):
    def find_best_c(num_steps, experiment_objs):
        removed_list = []
        num_steps_avg = np.mean(num_steps, axis=1)
        for counter1, i in enumerate(experiment_objs):
            for counter2, j in enumerate(experiment_objs):
                if i.num_iteration == j.num_iteration and \
                   i.num_simulation == j.num_simulation and \
                   i.simulation_depth == j.simulation_depth and \
                   i.tau == j.tau and \
                   i.c != j.c:
                    if num_steps_avg[counter1] > num_steps_avg[counter2]:
                        removed_list.append(counter1)
                    else:
                        removed_list.append(counter2)
        num_steps = np.delete(num_steps, removed_list, 0)
        experiment_objs_new = []
        for i, obj in enumerate(experiment_objs):
            if i not in removed_list:
                experiment_objs_new.append(obj)

        return num_steps, experiment_objs_new
    def find_best_tau(num_steps, experiment_objs):
        removed_list = []
        num_steps_avg = np.mean(num_steps, axis=1)
        for counter1, i in enumerate(experiment_objs):
            for counter2, j in enumerate(experiment_objs):
                if i.num_iteration == j.num_iteration and \
                   i.num_simulation == j.num_simulation and \
                   i.simulation_depth == j.simulation_depth and \
                   i.tau != j.tau:
                    if num_steps_avg[counter1] > num_steps_avg[counter2]:
                        removed_list.append(counter1)
                    else:
                        removed_list.append(counter2)
        num_steps = np.delete(num_steps, removed_list, 0)
        experiment_objs_new = []
        for i, obj in enumerate(experiment_objs):
            if i not in removed_list:
                experiment_objs_new.append(obj)

        return num_steps, experiment_objs_new

    num_steps, experiment_objs = find_best_c(num_steps, experiment_objs)
    print(num_steps.shape)
    num_steps, experiment_objs = find_best_tau(num_steps, experiment_objs)
    print(num_steps.shape)
    names = experiment_obj_to_name(experiment_objs)
    fig, axs = plt.subplots(1, 1, constrained_layout=False)

    for i, name in enumerate(names):
        num_steps_avg = np.mean(num_steps[i], axis=0)
        num_steps_std = np.mean(num_steps[i], axis=0)
        x = range(len(num_steps_avg))
        if len(num_steps_avg) == 1:
            color = generate_hex_color()
            axs.axhline(num_steps_avg, label=name, color=color)
            # axs.axhspan(num_steps_avg - 0.1 * num_steps_std,
            #             num_steps_avg + 0.1 * num_steps_std,
            #             alpha=0.4, color=color)

        else:
            axs.plot(x, num_steps_avg, label=name)
            axs.fill_between(x,
                             num_steps_avg - 0.1 * num_steps_std,
                             num_steps_avg + 0.1 * num_steps_std,
                             alpha=.4, edgecolor='none')
        axs.legend()
        fig.savefig(saved_name+".png")


def experiment_obj_to_name(experiment_objs):
    def all_elements_equal(List):
        result = all(element == List[0] for element in List)
        if (result):
            return True
        else:
            return False

    names = [""] * len(experiment_objs)
    print(names)
    agent_class = [i.agent_class for i in experiment_objs]
    if not all_elements_equal(agent_class):
        for i in range(len(experiment_objs)):
            names[i] += "agent:" + agent_class[i].name

    pre_trained = [i.pre_trained for i in experiment_objs]
    if not all_elements_equal(pre_trained):
        for i in range(len(experiment_objs)):
            names[i] += "pre_trained:" + str(pre_trained[i])

    model = [i.model for i in experiment_objs]
    if not all_elements_equal(model):
        for i in range(len(experiment_objs)):
            names[i] += "model:" + str(model[i])

    model_step_size = [i.model_step_size for i in experiment_objs]
    if not all_elements_equal(model_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "m_stepsize:" + str(model_step_size[i])

    # dqn params
    vf_network = [i.vf_network for i in experiment_objs]
    if not all_elements_equal(vf_network):
        for i in range(len(experiment_objs)):
            names[i] += "vf:" + str(vf_network[i])

    vf_step_size = [i.vf_step_size for i in experiment_objs]
    if not all_elements_equal(vf_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "vf_stepsize:" + str(vf_step_size[i])

    # mcts params
    c = [i.c for i in experiment_objs]
    if not all_elements_equal(c):
        for i in range(len(experiment_objs)):
            names[i] += "c:" + str(c[i])

    num_iteration = [i.num_iteration for i in experiment_objs]
    if not all_elements_equal(num_iteration):
        for i in range(len(experiment_objs)):
            names[i] += "N_I:" + str(num_iteration[i])

    simulation_depth = [i.simulation_depth for i in experiment_objs]
    if not all_elements_equal(simulation_depth):
        for i in range(len(experiment_objs)):
            names[i] += "S_D:" + str(simulation_depth[i])

    num_simulation = [i.num_simulation for i in experiment_objs]
    if not all_elements_equal(num_simulation):
        for i in range(len(experiment_objs)):
            names[i] += "N_S:" + str(num_simulation[i])

    model_corruption = [i.model_corruption for i in experiment_objs]
    if not all_elements_equal(model_corruption):
        for i in range(len(experiment_objs)):
            names[i] += "M_C:" + str(model_corruption[i])

    tau = [i.tau for i in experiment_objs]
    if not all_elements_equal(tau):
        for i in range(len(experiment_objs)):
            names[i] += "Tau:" + str(tau[i])
    print(names)
    return names


def generate_hex_color():
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    c = (r, g, b)
    hex_c = '#%02x%02x%02x' % c
    return hex_c