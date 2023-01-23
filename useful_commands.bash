python3 Main.py --env two_way_icy --scenario online --file_name "UAMCTS_Icy_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 1 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

python3 Main.py --env two_way --scenario online --file_name "UAMCTS_TwoWayW_Run0" \
--selection --expansion --simulation \
--backpropagation --num_run 5 \
--num_episode 300 --ni 10 --ns 10 --ds 30 \
--c 1 --tau 10

conda activate envp37

python3 Main.py --env two_way --scenario online --file_name "TEST" \
--selection --expansion --simulation \
--backpropagation --num_run 1 \
--num_episode 300 --ni 2 --ns 2 --ds 1 \
--c 1 --tau 10

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