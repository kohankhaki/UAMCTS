import pickle
import numpy as np


def combine_experiment_result(runs_dir, runs_list, result_dir, result_file_name):
    res = {'num_steps': None, 'rewards': None, 'experiment_objs': None, 'detail': None}
    for file_name in runs_list:
        print('hello')
        with open(runs_dir + file_name, 'rb') as f:
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
        # if res['experiment_objs'] is None:
        #     res['experiment_objs'] = result['experiment_objs']
        print(res['rewards'].shape, res['num_steps'].shape)
    with open(result_dir + result_file_name + '.p', 'wb') as f:
        pickle.dump(res, f)

def extract_experiment_result(input_dir, input_file_name, index, result_dir, result_file_name):
    res = {'num_steps': None, 'rewards': None, 'experiment_objs': None, 'detail': None}
    with open(input_dir + input_file_name, 'rb') as f:
        result = pickle.load(f)
    f.close()
    print(result['rewards'].shape, result['num_steps'].shape)
    res['num_steps'] = np.expand_dims(result['num_steps'][index], axis=0)
    res['rewards'] = np.expand_dims(result['rewards'][index], axis=0)
    print(res['rewards'].shape, res['num_steps'].shape)
    with open(result_dir + result_file_name + '.p', 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':

    runs_dir = "Results/"
    runs_list = [
        "UAMCTS_TwoWayIcyV2_ParamStudy_Run0.p",
        "UAMCTS_TwoWayIcyV2_ParamStudy_Run1.p",
        "UAMCTS_TwoWayIcyV2_ParamStudy_Run2.p",
        "UAMCTS_TwoWayIcyV2_ParamStudy_Run3.p",
        "UAMCTS_TwoWayIcyV2_ParamStudy_Run4.p",

        ]
    result_dir = "CombinedResults/"
    result_file_name = "UAMCTS_TwoWayIcyV2_ParamStudy"
    combine_experiment_result(runs_dir, runs_list, result_dir, result_file_name)


    # input_dir = "Results/"
    # input_file_name = "MCTS_TwoWayIcy_TrueModel_ParamStudy_Run0.p"
    # index = 0
    # result_dir = "CombinedResults/"
    # result_file_name = "MCTS_TwoWayIcy_TrueModel"
    # extract_experiment_result(input_dir, input_file_name, index, result_dir, result_file_name)