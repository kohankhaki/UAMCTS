import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pickle
import utils as utils
import random
import matplotlib.pyplot as plt
import os
import config
class UncertaintyNN(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(UncertaintyNN, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        state_size = state_shape[1]
        action_size = action_shape

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                self.add_module('hidden_layer_' + str(i), layer)
                self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")
        if len(layers_type) > 0:
            self.head = nn.Linear(layers_features[-1], 1)
        else:
            self.head = nn.Linear(state_size + action_size, 1)

    def forward(self, state, action):
        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.layers[i](x.float())
                x = torch.tanh(x)
            else:
                raise ValueError("layer is not defined")
        if len(self.layers_type) > 0:
            head = self.head(x.float())
        else:
            x = state.flatten(start_dim= 1)
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)
            head = self.head(x.float())
        return head


def create_minimal_dataset(buffer):
    new_buffer = []
    for data in buffer:
        prev_state_pos_onehot = getOnehotTorch(torch.tensor([data.prev_state[0][-6]], dtype=int).unsqueeze(0), 10)
        prev_state_shottimer_onehot = getOnehotTorch(torch.tensor([data.prev_state[0][-1]], dtype=int).unsqueeze(0), 5)
        minimal_prev_state = torch.cat((prev_state_shottimer_onehot, prev_state_pos_onehot), dim=1)
        minimal_prev_state = torch.tensor([data.prev_state[0][-6], data.prev_state[0][-1]]).unsqueeze(0)
        transition = utils.corrupt_transition(minimal_prev_state,
                                            data.prev_action,
                                            data.true_state,
                                            data.corrupt_state)
        new_buffer.append(transition)
    
    with open("MiniAtariResult/Buffer/SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MinimalOnehotBuffer149620_t149600.p", 'wb') as file:
        pickle.dump(new_buffer, file)

def create_onehot_dataset(buffer):
    new_buffer = []
    for data in buffer:
        #space invaders ***
        # prev_state_pos_onehot = getOnehotTorch(torch.tensor([data.prev_state[0][-6]], dtype=int).unsqueeze(0), 10)
        # prev_state_shottimer_onehot = getOnehotTorch(torch.tensor([data.prev_state[0][-1]], dtype=int).unsqueeze(0), 5)
        # one_hot_prev_state = torch.cat((data.prev_state[0][0:-6], data.prev_state[0][-5:-1])).unsqueeze(0)
        # one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_pos_onehot), dim=1)
        # one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_shottimer_onehot), dim=1)
        #***

        #freeway ***
        state_copy = torch.clone(data.prev_state)
        # one_hot_prev_state = getOnehotTorch(torch.tensor([state_copy[0][-2]], dtype=int).unsqueeze(0), 10)
        # one_hot_prev_state = torch.tensor([])
        # state_copy[0][3] += 5
        # state_copy[0][7] += 5
        # state_copy[0][11] += 5
        # state_copy[0][15] += 5
        # state_copy[0][19] += 5
        # state_copy[0][23] += 5
        # state_copy[0][27] += 5
        # state_copy[0][31] += 5
        # for i in range(state_copy.shape[1]):
        #     if i % 4 == 3:
        #         s = getOnehotTorch(torch.tensor([state_copy[0][i]], dtype=int).unsqueeze(0), 11)
        #     else:        
        #         s = getOnehotTorch(torch.tensor([state_copy[0][i]], dtype=int).unsqueeze(0), 10)
        #     one_hot_prev_state = torch.cat((one_hot_prev_state, s), dim=1)
        one_hot_prev_state = state_copy[0, -20:-10].unsqueeze(0)
        # ***
        transition = utils.corrupt_transition(one_hot_prev_state,
                                            data.prev_action,
                                            data.true_state,
                                            data.corrupt_state)
        # if (np.array_equal(data.prev_action[0], [1, 0, 0])):
        new_buffer.append(transition)
    return new_buffer
    # with open("MiniAtariResult/Buffer/Freeway_CorruptedStates=[1, 2, 3, 4, 5, 6, 7]_OnehotBuffer4799_t4500.p", 'wb') as file:
    #     pickle.dump(new_buffer, file)

def train_uncertainty(corrupt_transition_batch, network, optimizer, max_uncertainty, min_uncertainty):
    batch = utils.corrupt_transition(*zip(*corrupt_transition_batch))
    true_next_states = torch.cat([s for s in batch.true_state]).float()
    corrupt_next_states = torch.cat([s for s in batch.corrupt_state]).float()
    prev_state_batch = torch.cat(batch.prev_state).float()
    prev_action_batch = torch.cat(batch.prev_action).float()
    prev_state_batch = prev_state_batch.unsqueeze(1) #only pos
    # print(prev_state_batch, prev_action_batch)
    predicted_uncertainty = network(prev_state_batch, prev_action_batch)
    true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1).unsqueeze(1) #/ max_uncertainty
    loss = F.mse_loss(predicted_uncertainty,
                        true_uncertainty)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_uncertainty(corrupt_transition_batch, network):
    with torch.no_grad():
        batch = utils.corrupt_transition(*zip(*corrupt_transition_batch))
        true_next_states = torch.cat([s for s in batch.true_state]).float()
        corrupt_next_states = torch.cat([s for s in batch.corrupt_state]).float()
        prev_state_batch = torch.cat(batch.prev_state).float()
        prev_action_batch = torch.cat(batch.prev_action).float()
        prev_state_batch = prev_state_batch.unsqueeze(1) #only pos
        predicted_uncertainty = network(prev_state_batch, prev_action_batch)
        true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1).unsqueeze(1)
        loss = torch.mean((predicted_uncertainty - true_uncertainty)**2, dim=0)
    return loss.item()

def one_hot_to_int(one_hot):
    index = 0 
    for i in one_hot:
        if int(i.item()) == 1:
            return index
        index += 1

def draw_data(data):
    data = utils.corrupt_transition(*zip(*data))
    prev_action = torch.cat(data.prev_action)
    prev_state = torch.cat(data.prev_state)

    action_index = []
    counter = np.zeros([3, 10])
    for i in range(len(prev_action)):
        if np.array_equal(prev_action[i], [1, 0, 0]):
            action_index.append(i)
            pos = one_hot_to_int(prev_state[i, -20:-10])
            counter[0, pos] += 1
        if np.array_equal(prev_action[i], [0, 1, 0]):
            action_index.append(i)
            pos = one_hot_to_int(prev_state[i, -20:-10])
            counter[1, pos] += 1
        if np.array_equal(prev_action[i], [0, 0, 1]):
            action_index.append(i)
            pos = one_hot_to_int(prev_state[i, -20:-10])
            counter[2, pos] += 1
    print(counter)
    exit(0)

    prev_state = prev_state[:, -2]
    true_next_states = torch.cat([s for s in data.true_state]).float()
    corrupt_next_states = torch.cat([s for s in data.corrupt_state]).float()

    prev_state = prev_state[action_index]
    true_next_states = true_next_states[action_index]
    corrupt_next_states = corrupt_next_states[action_index]
    print(prev_state.shape, true_next_states.shape, corrupt_next_states.shape)
    true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1)
    # for i in range(len(true_uncertainty)):
    #     if true_uncertainty[i] > 1 and prev_state[i] == 1:
    #         print(prev_state[i])
    #         print(true_next_states[i], "\n", corrupt_next_states[i], prev_state[i])
    #         exit(0)
    
    # x = true_uncertainty[prev_state==1]
    # corrupt_next_states = corrupt_next_states[prev_state==1]
    # true_next_states = true_next_states[prev_state==1]
    # prev_state2 = torch.cat(data.prev_state)[action_index]
    # prev_state = prev_state2[prev_state == 1]
    # print(corrupt_next_states[x > 0], true_next_states[x > 0], prev_state[x > 0])

    plt.scatter(prev_state, true_uncertainty)
    plt.savefig("buffer.png")
    print(prev_state.shape, true_uncertainty.shape)

def unique_datapoints(data):
    # print(data[0])
    # exit(0)
    data = utils.corrupt_transition(*zip(*data))
    prev_state = torch.cat(data.prev_state).numpy()
    true_next_states = torch.cat(data.true_state).numpy()
    corrupt_next_states = torch.cat(data.corrupt_state).numpy()
    true_uncertainty = np.mean((true_next_states - corrupt_next_states) ** 2, axis=1)
    print(true_uncertainty.shape, np.count_nonzero(true_uncertainty) / prev_state.shape[0])

    unique_prev_state = np.unique(prev_state, axis=0)
    print("buffer size:", prev_state.shape, "unique points:", unique_prev_state.shape)
    return unique_prev_state.shape[0]

def save_uncertainty(name, uncertainty_model):
    with open("MiniAtariResult/Paper/SavedUncertainty/"+name, 'wb') as file:
        pickle.dump(uncertainty_model, file)

def load_uncertainty(name):
    with open("MiniAtariResult/Paper/SavedUncertainty/"+name, 'rb') as file:
        uncertainty_model = pickle.load(file)
    return uncertainty_model

def getOnehotTorch(index, num_actions):
    '''
    action = index torch
    output = onehot torch
    '''
    batch_size = index.shape[0]
    onehot = torch.zeros([batch_size, num_actions], device="cpu")
    # onehot.zero_()
    onehot.scatter_(1, index, 1)
    return onehot

def draw_loss(loss_lists_dir):
    loss_files = os.listdir(loss_lists_dir)
    for loss_file in loss_files:
        # print(loss_file)
        with open(loss_lists_dir+"/"+loss_file, 'rb') as file:
            loss = pickle.load(file)
        label = loss_file.split("]")[1].split(".")[0]
        plt.plot(loss, label=label)
        try:
            print(loss[-1], label)
        except:
            print(loss, label)
        plt.yscale("log")
    plt.legend()
    plt.savefig("loss.png")

def check_uncertainty(data, network):
    data = utils.corrupt_transition(*zip(*data))
    prev_action = torch.cat(data.prev_action)
    prev_state = torch.cat(data.prev_state)
    true_next_states = torch.cat([s for s in data.true_state]).float()
    corrupt_next_states = torch.cat([s for s in data.corrupt_state]).float()
    counter = np.zeros([4, 10])
    sum = np.zeros([4, 10])
    true = np.zeros([4, 10])
    mse = 0
    for i in range(len(prev_action)):
        if prev_state.shape[1] == 348:
            pos = one_hot_to_int(prev_state[i, -20:-10])
        elif prev_state.shape[1] == 155:
            pos = one_hot_to_int(prev_state[i, 125:135])
        elif prev_state.shape[1] == 319:
            pos = one_hot_to_int(prev_state[i, -15:-5])

        with torch.no_grad():
            predicted_uncertainty = network(prev_state[i].unsqueeze(0), prev_action[i].unsqueeze(0)).item()
        true_uncertainty = torch.mean((true_next_states[i] - corrupt_next_states[i]) ** 2).item()
        mse += (predicted_uncertainty - true_uncertainty) ** 2
        '''
        if np.array_equal(prev_action[i], [1, 0, 0]):
            counter[0, pos] += 1
            sum[0, pos] += predicted_uncertainty
            true[0, pos] += true_uncertainty
        elif np.array_equal(prev_action[i], [0, 1, 0]):
            counter[1, pos] += 1
            sum[1, pos] += predicted_uncertainty
            true[1, pos] += true_uncertainty
        elif np.array_equal(prev_action[i], [0, 0, 1]):
            counter[2, pos] += 1
            sum[2, pos] += predicted_uncertainty
            true[2, pos] += true_uncertainty
        elif np.array_equal(prev_action[i], [1, 0, 0, 0]):
            counter[0, pos] += 1
            sum[0, pos] += predicted_uncertainty
            true[0, pos] += true_uncertainty
        elif np.array_equal(prev_action[i], [0, 1, 0, 0]):
            counter[1, pos] += 1
            sum[1, pos] += predicted_uncertainty
            true[1, pos] += true_uncertainty
        elif np.array_equal(prev_action[i], [0, 0, 1, 0]):
            counter[2, pos] += 1
            sum[2, pos] += predicted_uncertainty
            true[2, pos] += true_uncertainty
        elif np.array_equal(prev_action[i], [0, 0, 0, 1]):
            counter[3, pos] += 1
            sum[3, pos] += predicted_uncertainty
            true[3, pos] += true_uncertainty
        '''
    # print(sum/counter)
    # print("***")
    # print(true/counter)
    return (mse ** 0.5) / len(prev_action)
    # for param in network.parameters():
    #     print(param.data)

def main_train(buffer, num_epochs, name):
    uf = {'network':None,
            'batch_size': 32,
            'step_size':0.001,
            'layers_type':['fc', 'fc'],
            'layers_features':[32, 32],
            'training':True}

    action_shape = buffer[0].prev_action.shape[1]
    uf['network'] = UncertaintyNN(buffer[0].prev_state.shape, action_shape, uf['layers_type'], uf['layers_features'])

    optimizer = optim.Adam(uf['network'].parameters(), lr=uf['step_size'])
    show_freq = 299
    save_freq = 500
    true_uncertainty_list = []
    for i in range(len(buffer)):
        true_uncertainty = torch.mean(torch.pow(buffer[i].true_state.float() - buffer[i].corrupt_state.float(), 2)).item()
        true_uncertainty_list.append(true_uncertainty)
    max_uncertainty = max(true_uncertainty_list)
    min_uncertainty = min(true_uncertainty_list)

    loss_list = []
    for i in range(num_epochs):
        batch = random.sample(buffer, k=uf['batch_size'])
        train_uncertainty(batch, uf['network'], optimizer, max_uncertainty, min_uncertainty)
        loss = test_uncertainty(buffer, uf['network'])
        print(i, ":", loss)
        loss_list.append(loss)
        if (i+1) % save_freq == 0:
            with open("MiniAtariResult/Thesis/UncertaintyModel/epoch="+str(i)+name, 'wb') as file:
                pickle.dump(uf, file)
        # if i % 500 == 499:
        #     with open("MiniAtariResult/Loss/SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]"+"len_onehot_buffer_16x8_e"+str(i)+"="+str(len(buffer))+".p", 'wb') as file:
        #         pickle.dump(loss_list, file)
    return loss_list

def draw_multiple_loss():
    all_files = os.listdir("MiniAtariResult/Thesis/UncertaintyModel")
    loss_runs = []
    for file in all_files:
        if "Loss" in file and "Freeway" in file and "=7000" in file:
            with open("MiniAtariResult/Thesis/UncertaintyModel/"+file, 'rb') as file:
                loss = pickle.load(file)
            loss_runs.append(loss)
    loss_runs = np.asarray(loss_runs)
    avg_loss = np.mean(loss_runs, axis=0)
    std_loss = np.std(loss_runs, axis=0)
    x = np.arange(loss_runs.shape[1])
    plt.yscale("log")
    plt.plot(x, avg_loss)
    plt.fill_between(x,
                    avg_loss - std_loss,
                    avg_loss + std_loss,
                    alpha=.4, edgecolor='none')
    plt.savefig("train_uncertainty.png")
def main_check(name):
    if name == "space":
        # 1.607954439655965e-05
        buffers = Config.space_buffer
        uncertainties = Config.space_uncertainty
    elif name == "breakout":
        # 3.845056750670959e-05
        buffers = Config.breakout_buffer
        uncertainties = Config.breakout_uncertainty
    elif name == "freeway":
        # 1.7065088796630053e-05
        buffers = Config.freeway_buffer
        uncertainties = Config.freeway_uncertainty
    print(len(buffers), len(uncertainties))
    sum_errors = 0
    for i, u_name in enumerate(uncertainties):
        b_name = buffers[i // 3]
        with open("MiniAtariResult/Paper/Buffer/"+b_name+".p", 'rb') as file:
            buffer = pickle.load(file)
        with open("MiniAtariResult/Paper/SavedUncertainty/"+u_name, 'rb') as file:
            uf = pickle.load(file)    
        err = check_uncertainty(buffer, uf['network'])
        print(i, err)
        sum_errors += err
    return sum_errors / len(uncertainties)

if __name__ == "__main__":
    # draw_multiple_loss()
    # exit(0)
    # print(main_check("breakout"))

    buffer_folder = "MiniAtariResult/Thesis/Buffer/Breakout2"
    buffer_file_name = ['/home/farnaz/UAMCTS/Results/UncertaintyBuffers/l=24500_e=999__r=0_3_Breakout_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run1.p',
'/home/farnaz/UAMCTS/Results/UncertaintyBuffers/l=15292_e=999__r=0_2_Freeway_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run0.p',
'/home/farnaz/UAMCTS/Results/UncertaintyBuffers/l=82079_e=299__r=0_SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run3.p'
    ]




    # unique_data = 0
    # for f in buffer_file_name:
    #     with open(buffer_folder+"/"+f, 'rb') as file:
    #         buffer = pickle.load(file)
    #     buffer = buffer[0: 3000]
    #     unique_data += unique_datapoints(buffer)
    # print(unique_data / (1000 * len(buffer_file_name)) )
    # exit(0)
    #train U
    for f in buffer_file_name:
        with open(f, 'rb') as file:
            buffer = pickle.load(file)
        # buffer = buffer[0: 1000]
        unique_datapoints(buffer)
    exit(0)
        # name = "bsize="+str(len(buffer))+f.split("MCTS")[0][:-1]+".p"
        # loss_list = main_train(buffer, num_epochs=10000, name=name)
        # with open("MiniAtariResult/Thesis/UncertaintyModel/Loss_"+name, 'wb') as file:
        #     pickle.dump(loss_list, file)
        # plt.plot(loss_list)
        # plt.savefig("uncertainty_train.png")


    # save_uncertainty(buffer_file_name, uf)    
    # with open("MiniAtariResult/Paper/SavedUncertainty/"+uncertainty_name, 'rb') as file:
   #     uf = pickle.load(file)              
    # check_uncertainty(buffer, uf['network'])
