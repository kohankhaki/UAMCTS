################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 dqn.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import numpy as np
import random, numpy, argparse, logging, os

from collections import namedtuple
from minatar import Environment

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
start_run_number = 20
finish_run_number = 30 
env_name = "freeway" #space_invaders, freeway, breakout
folder_name = "DQN_Freeway/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StateActionVFNN(nn.Module):
    def __init__(self, state_shape, num_actions, layers_type, layers_features, action_layer_num):
        # state : Batch, Linear State
        # action: Batch, A
        super(StateActionVFNN, self).__init__()
        self.layers_type = layers_type
        self.action_layer_num = action_layer_num
        self.layers = []
        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                action_shape_size = 0
                if i == self.action_layer_num:
                    # insert action to this layer
                    action_shape_size = num_actions

                if i == 0:
                    linear_input_size = state_shape[1] + action_shape_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                    # nn.init.normal_(layer.weight)
                    self.add_module('hidden_layer_' + str(i), layer)
                    self.layers.append(layer)

                else:
                    layer = nn.Linear(layers_features[i - 1] + action_shape_size, layers_features[i])
                    # nn.init.normal_(layer.weight)
                    self.add_module('hidden_layer_' + str(i), layer)
                    self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        if len(layers_type) > 0:
            if self.action_layer_num == len(self.layers_type):
                self.head = nn.Linear(layers_features[-1] + num_actions, 1)
                # nn.init.normal_(self.head.weight)

            elif self.action_layer_num == len(self.layers_type) + 1:
                self.head = nn.Linear(layers_features[-1], num_actions)
                # nn.init.normal_(self.head.weight)

            else:
                self.head = nn.Linear(layers_features[-1], 1)
                # nn.init.normal_(self.head.weight)
        else:
            # simple linear regression
            if self.action_layer_num == len(self.layers_type):
                self.head = nn.Linear(state_shape[1] + num_actions, 1, bias=False)
            elif self.action_layer_num == len(self.layers_type) + 1:
                self.head = nn.Linear(state_shape[1], num_actions, bias=False)
                # nn.init.normal_(self.head.weight)

    def forward(self, state, action=None):
        if self.action_layer_num != len(self.layers) + 1 and action is None:
            raise ValueError("action is not given")
        x = state.flatten(start_dim=1)
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim=1)
                if i == self.action_layer_num:
                    # insert action to this layer
                    a = action.flatten(start_dim=1)
                    x = torch.cat((x.float(), a.float()), dim=1)
                x = self.layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")

        if self.action_layer_num == len(self.layers_type):
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)

        x = self.head(x.float())
        return x

transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)



def get_state(s):
    if env_name == "space_invaders":
        tmp = np.append(np.append(s[1].flatten(), s[2].flatten()), s[3].flatten())
        s = np.append(np.append(
            np.append(np.append(np.append(np.append(tmp, s[0]), s[4]), s[5]), s[6]),
            s[7]), s[8])

    elif env_name == "freeway":
        s = np.append(np.append(np.asarray(s[1]).flatten(), s[0]), s[2])

    elif env_name == "breakout":
        s = np.append(np.append(np.append(np.append(np.append(np.append(
            np.append(s[0], s[1])
            , s[2]), s[3]), s[4]), s[5])
            , s[6]), s[7].flatten())

    else:
        raise ValueError("env not defined!")
    return torch.tensor(s, device=device).unsqueeze(0).float()


def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net):

    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                action = policy_net(s).max(1)[1].view(1, 1)

    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(env.minimal_action_set()[action])

    # Obtain s_prime
    # s_prime = get_state(env.state())
    s_prime = get_state(env.game_state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)



def train(sample, policy_net, target_net, optimizer):
    batch_samples = transition(*zip(*sample))

    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)

   
    Q_s_a = policy_net(states).gather(1, actions)
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)
    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        Q_s_prime_a_prime[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

    # Compute the target
    target = rewards + GAMMA * Q_s_prime_a_prime

    # Huber loss
    loss = f.smooth_l1_loss(target, Q_s_a)

    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def dqn(env, replay_off, target_off, output_file_name, store_intermediate_result=True, load_path=None, step_size=STEP_SIZE):
    store_intermediate_result = True
    # Get channels and number of actions specific to each game
    action_list = env.minimal_action_set()
    num_actions = len(action_list)#env.num_actions()
    if env_name == "space_invaders":
        state_shape = [1, 306]
    elif env_name == "freeway":
        state_shape = [1, 34]
    elif env_name == "breakout":
        state_shape = [1, 107]
    else:
        raise ValueError("env name not defined!")
    # Instantiate networks, optimizer, loss and buffer
    # policy_net = QNetwork(in_channels, num_actions).to(device)
    ValueFunction = {'q': dict(network=None,
                    layers_type=['fc', 'fc'],
                    layers_features=[64, 64],
                    action_layer_num=3,
                    batch_size=32,
                    step_size=0.00025,
                    training=False)}
    policy_net = StateActionVFNN(state_shape, num_actions, 
                        ValueFunction['q']['layers_type'], 
                        ValueFunction['q']['layers_features'], 
                        ValueFunction['q']['action_layer_num']).to(device)
    ValueFunction['q']['network'] = policy_net
    replay_start_size = 0
    if not target_off:
        # target_net = QNetwork(in_channels, num_actions).to(device)
        target_net = StateActionVFNN(state_shape, num_actions, 
                        ValueFunction['q']['layers_type'], 
                        ValueFunction['q']['layers_features'], 
                        ValueFunction['q']['action_layer_num']).to(device)

        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        if not target_off:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if not replay_off:
            r_buffer = checkpoint['replay_buffer']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init = checkpoint['episode']
        t_init = checkpoint['frame']
        policy_net_update_counter_init = checkpoint['policy_net_update_counter']
        avg_return_init = checkpoint['avg_return']
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Set to training mode
        policy_net.train()
        if not target_off:
            target_net.train()

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        # s = get_state(env.state())
        s = get_state(env.game_state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, policy_net)

            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    train(sample, policy_net, policy_net, optimizer)
                else:
                    policy_net_update_counter += 1
                    train(sample, policy_net, target_net, optimizer)

            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)
        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 1000 == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )
            print("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1000 == 0:
            with open("MiniAtariResult/Thesis/ValueFunction/" + output_file_name + "_E"+str(e)+"_64x64_VF.p", "wb") as file:
                pickle.dump(ValueFunction, file)
            with open("MiniAtariResult/Thesis/Results/" + folder_name + output_file_name +"_E"+str(e)+'_DQN_64x64_Return.p', 'wb') as f:
                result = {'rewards': data_return}
                pickle.dump(result, f)
        if store_intermediate_result and t % 500000 == 0:
            with open("MiniAtariResult/Thesis/ValueFunction/" + output_file_name + "_F"+str(t)+"_64x64_VF.p", "wb") as file:
                pickle.dump(ValueFunction, file)
            with open("MiniAtariResult/Thesis/Results/"+ folder_name + output_file_name +"_F"+str(t)+'_DQN_64x64_Return.p', 'wb') as f:
                result = {'rewards': data_return}
                pickle.dump(result, f)
            

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))
        


    with open("MiniAtariResult/Thesis/ValueFunction/" + output_file_name +"_E"+str(e)+"_64x64_VF.p", "wb") as file:
                pickle.dump(ValueFunction, file)
    with open("MiniAtariResult/Thesis/Results/"+ folder_name + output_file_name +"_E"+str(e)+'_DQN_64x64_Return.p', 'wb') as f:
        result = {'rewards': data_return}
        pickle.dump(result, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)


    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile
    print("env:", env_name)
    env = Environment(env_name)
    print('Cuda available?: ' + str(torch.cuda.is_available()))
    for i in range(start_run_number, finish_run_number):
        file_name = env_name + "_run"+str(i)
        dqn(env, args.replayoff, args.targetoff, file_name, args.save, load_file_path, args.alpha)


if __name__ == '__main__':
    main()

