s_vf_list = [0.01]
s_md_list = [0.1]
model_corruption_list = [0]
model_list = [{'type': 'heter', 'layers_type': ['fc'], 'layers_features': [6], 'action_layer_num': 2}]
trained_vf_list = [None]

experiment_detail = ""

rollout_idea = 5 # None, 1, 5
selection_idea = 1  # None, 1
backpropagate_idea = 1  # None, 1
expansion_idea = 2 # None, 2

pre_gathered_buffer = None 

num_runs = 5
num_episode = 1 #spc=300 frw=1500 brk=2000
max_step_each_episode = 300

u_batch_size = 32
minimum_uncertainty_buffer_training = u_batch_size
u_step_size = 0.001
u_layers_type = ['fc', 'fc']
u_layers_features = [128, 128]
u_training = True
u_epoch_training = 5000
u_epoch_training_rate = 5000

u_training_steps = [u_epoch_training_rate * i for i in range(num_episode * max_step_each_episode // u_epoch_training_rate)]

u_pretrained_u_network = None
use_perfect_uncertainty = True

env_name = "breakout" #freeway, breakout, space_invaders

c_list = [2 ** 0.5] 
num_iteration_list = [2] #[10] [100]
simulation_depth_list = [2] #[20] [50]
num_simulation_list = [2] #[10] 
tau_list = [0.1]

save_uncertainty_buffer = False