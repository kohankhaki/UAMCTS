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
python3 Analyze.py --scenario online --file_name UAMCTS_TwoWayIcy_ParamStudy.p --plot_name UAMCTS_TwoWayIcyParamStudy --metric num_steps

#test
python3 Main.py --env two_way --scenario online --file_name "test" \
--selection --expansion --simulation \
--backpropagation --num_run 1 \
--num_episode 300 --ni 1 --ns 1 --ds 1 \
--c 1 --tau 10
