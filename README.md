# Monte Carlo Tree Search in the Presence of Model Uncertainty
This is the readme file for running the experiments in the *Monte Carlo Tree Search in the Presence of Model Uncertainty* paper.

## Libraries requirements
You need to have installed Python 3.8. You also need to install PyTorch, NumPy, Matplotlib, torchvision, tqdm packages:

```
pip3 install torch torchvision torchaudio
pip3 install matplotlib
pip3 install tqdm
```
You also need to install the **modified** version of the [MinAtar Environments](https://github.com/kenjyoung/MinAtar) than includes the corrupted models. To install the MinAtar Environments, use the following command in terminal:
```
pip install MinAtar/.
```

## Experiments
To run the experiments, use the following command:

```
Main.py [-h] --env ENV --scenario SCENARIO --file_name FILE_NAME
               [--selection] [--expansion] [--simulation]
               [--backpropagation] [--num_run NUM_RUN]
               [--num_episode NUM_EPISODE] [--ni NI] [--ns NS] [--ds DS]
               [--c C] [--tau TAU] [--learn_transition] [--use_true_model]
```
- ENV can be "space_invaders", "freeway", "breakout, or "two_way"
- SCENARIO can be "online" or "offline"
- FILE_NAME is the name of the file of the result of the experiments. This file contians a dictionary with keys 'num_steps' and 'rewards'.
- Use any of the commands --selection, --expansion, --simulation, or --backpropagation to active the the UA corresponding component.
- NUM_RUN and NUM_EPISODE are the number of runs and episodes.
- NI, NS, and DS are the number of iterations, number of simulations, and depth o simulations respectively.
- C is exploration constant and TAU is uncertainty factor. For the online scenario of UAMCTS, TAU is the initial value of the uncertainty factor. 
- Use use_true_model option if you want the agent to have access to the true model of the environment.
- Use learn_transition option if you want to run an experiment with the MCTS agent that learns the transition function online. This option works only for the "two_way" environment and not for the UA components.
- If you want to change the uncertainty function or the transition function networks' parameter, go to UAMCTS/config.py.



