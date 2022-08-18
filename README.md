#Monte Carlo Tree Search in the Presence of Model Uncertainty
This is the readme file for running the experiments in the *Monte Carlo Tree Search in the Presence of Model Uncertainty* paper.

##Libraries requirements
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


python Main.py --env two_way --scenario online --ni 2  --file_name tmp --selection --learn_transition