# Kiarash
python3 Main.py --env two_way --scenario online --file_name "UAMCTS_TwoWayW_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way --scenario online --file_name "UAMCTS_TwoWayW_Run1" \
--selection --expansion --simulation \
--backpropagation --num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way --scenario online --file_name "UAMCTS_TwoWayW_Run2" \
--selection --expansion --simulation \
--backpropagation --num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

#Kiarash 2
python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcy_ParamStudy_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcy_ParamStudy_Run1" \
--selection --expansion --simulation \
--backpropagation --num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

#Kiarash 3
python3 Main.py --env two_way_icy --scenario offline --file_name "MCTS_TwoWayIcy_ParamStudy_Run0" \
--num_run 100 --num_episode 1 --ni 10 --ns 10 --ds 30 

python3 Main.py --env two_way_icy --scenario offline --file_name "MCTS_TwoWayIcy_TrueModel_ParamStudy_Run0" \
--num_run 100 --num_episode 1 --ni 10 --ns 10 --ds 30 --use_true_model

#Kiarash 4
python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcy_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 25 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcy_Run1" \
--selection --expansion --simulation \
--backpropagation --num_run 25 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcy_Run2" \
--selection --expansion --simulation \
--backpropagation --num_run 25 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcy_Run3" \
--selection --expansion --simulation \
--backpropagation --num_run 25 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10


#farnaz
python3 Analyze.py --scenario offline --file_name MCTS_TwoWayIcy_TrueModel_ParamStudy_Run0.p --plot_name test --metric num_steps

#Kiarash 5
python3 Main.py --env two_way --scenario online --file_name "MCTS_Residual_H8_ParamStudy_Run0" \
--num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10 --learn_residual

python3 Main.py --env two_way --scenario online --file_name "MCTS_Residual_H8_ParamStudy_Run1" \
--num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10 --learn_residual

#Kiarash 6
python3 Main.py --env two_way_icy --scenario offline --file_name "MCTS_TwoWayIcyV2_ParamStudy_Run0" \
--num_run 10 --num_episode 1 --ni 10 --ns 10 --ds 30 

python3 Main.py --env two_way_icy --scenario offline --file_name "MCTS_TwoWayIcyV2_TrueModel_ParamStudy_Run0" \
--num_run 10 --num_episode 1 --ni 10 --ns 10 --ds 30 --use_true_model

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_ParamStudy_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 2 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_ParamStudy_Run1" \
--selection --expansion --simulation \
--backpropagation --num_run 2 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_ParamStudy_Run2" \
--selection --expansion --simulation \
--backpropagation --num_run 2 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_ParamStudy_Run3" \
--selection --expansion --simulation \
--backpropagation --num_run 2 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_ParamStudy_Run4" \
--selection --expansion --simulation \
--backpropagation --num_run 2 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

#kiarash 7
python3 Main.py --env two_way_icy --scenario offline --file_name "MCTS_TwoWayIcyV2" \
--num_run 30 --num_episode 1 --ni 10 --ns 10 --ds 30 --c 0.5

python3 Main.py --env two_way_icy --scenario offline --file_name "MCTS_TwoWayIcyV2_TrueModel" \
--num_run 30 --num_episode 1 --ni 10 --ns 10 --ds 30 --c 0.5 --use_true_model

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 10 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_Run1" \
--selection --expansion --simulation \
--backpropagation --num_run 10 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_TwoWayIcyV2_Run2" \
--selection --expansion --simulation \
--backpropagation --num_run 10 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

#kiarash 8
python3 Main.py --env two_way_icy --scenario offline --file_name "UAMCTS_TwoWayIcy_Offline_tau0d1" \
--selection --expansion --simulation \
--backpropagation --num_run 30 \
--num_episode 1 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 0.1

python3 Main.py --env two_way_icy --scenario offline --file_name "UAMCTS_TwoWayIcy_Offline_tau1" \
--selection --expansion --simulation \
--backpropagation --num_run 30 \
--num_episode 1 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 1

python3 Main.py --env two_way_icy --scenario offline --file_name "UAMCTS_TwoWayIcy_Offline_tau10" \
--selection --expansion --simulation \
--backpropagation --num_run 30 \
--num_episode 1 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10