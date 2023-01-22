rollout_idea = 5
selection_idea = 1 
backpropagate_idea = 1  
expansion_idea = 2

num_runs = 5
num_episode = 1
max_step_each_episode = 300

# Below you can change the parameters of the transition network. (from line 21 to 35)
# st_batch_size is the batch size.
st_batch_size = 32
# st_step_size is the step size of the optimizer
st_step_size = 0.001
# st_layers_type defines the type of the hidden layers in a list. 
# 'fc' defines a fully connected hidden layer.
# st_layers_features defines the number of the hidden units in each corresponding hidden layer. 
st_layers_type = ['fc']
st_layers_features = [8]
# st_epoch_training is the parameter E and st_epoch_training_rate is I
st_epoch_training = 10000
st_epoch_training_rate = 500
minimum_transition_buffer_training = st_batch_size
st_training = True
st_training_steps = [st_epoch_training_rate * i for i in range(num_episode * max_step_each_episode // st_epoch_training_rate)]

# Below you can change the parameters of the uncertainty network. (from line 38 to 42)
# u_batch_size is the batch size.
u_batch_size = 32
# u_step_size is the step size of the optimizer
u_step_size = 0.001
# u_layers_type defines the type of the hidden layers in a list. 
# 'fc' defines a fully connected hidden layer.
# u_layers_features defines the number of the hidden units in each corresponding hidden layer. 
u_layers_type = ['fc', 'fc']
u_layers_features = [128, 128]
# u_epoch_training is the parameter E and u_epoch_training_rate is I
u_epoch_training = 300
u_epoch_training_rate = 5000
minimum_uncertainty_buffer_training = u_batch_size
u_training = True
u_training_steps = [u_epoch_training_rate * i for i in range(num_episode * max_step_each_episode // u_epoch_training_rate)]

u_pretrained_u_network = None
use_perfect_uncertainty = False
save_uncertainty_buffer = False
pre_gathered_buffer = None 

experiment_detail = ""

s_vf_list = [0.01]
s_md_list = [0.1]
model_corruption_list = [0]
model_list = [{'type': 'heter', 'layers_type': ['fc'], 'layers_features': [6], 'action_layer_num': 2}]
trained_vf_list = [None]