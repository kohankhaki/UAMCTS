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

if __name__ == '__main__':

    runs_dir = "Results/"
    runs_list = [
        "UAMCTS_TwoWayW_Run0.p",
        "UAMCTS_TwoWayW_Run1.p",
        "UAMCTS_TwoWayW_Run2.p"
    ]
    result_dir = "CombinedResults/"
    result_file_name = "UAMCTS_TwoWayW"
    combine_experiment_result(runs_dir, runs_list, result_dir, result_file_name)