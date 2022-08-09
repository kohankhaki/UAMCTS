from numpy import MachAr

freeway_buffer = [
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1508_episode:90_run:0',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1472_episode:90_run:1',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1475_episode:90_run:2',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1424_episode:90_run:3',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1378_episode:90_run:4',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1516_episode:90_run:5',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1381_episode:90_run:6',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1495_episode:90_run:7',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1454_episode:90_run:8',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1468_episode:90_run:9',
    'Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1371_episode:90_run:10', 
]

space_buffer = [
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode:20_run:0',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode:20_run:1',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode:20_run:2',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode:20_run:3',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6290_episode:20_run:4',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode:20_run:5',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6283_episode:20_run:6',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=7',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=8',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=9'
]

breakout_buffer =[
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1630_episode:90_run:0",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1580_episode:90_run:1",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1406_episode:90_run:3",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1518_episode:90_run:4",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1622_episode:90_run:6",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1608_episode:90_run:8",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode:90_run:9",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1566_episode:90_run:11",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1434_episode:90_run:12",
    "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode:90_run:14",
]

freeway_uncertainty = [
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1508_episode=90_run=0_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1508_episode=90_run=0_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1508_episode=90_run=0_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1472_episode=90_run=1_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1472_episode=90_run=1_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1472_episode=90_run=1_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1475_episode=90_run=2_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1475_episode=90_run=2_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1475_episode=90_run=2_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1424_episode=90_run=3_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1424_episode=90_run=3_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1424_episode=90_run=3_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1378_episode=90_run=4_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1378_episode=90_run=4_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1378_episode=90_run=4_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1516_episode=90_run=5_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1516_episode=90_run=5_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1516_episode=90_run=5_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1381_episode=90_run=6_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1381_episode=90_run=6_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1381_episode=90_run=6_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1495_episode=90_run=7_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1495_episode=90_run=7_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1495_episode=90_run=7_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1454_episode=90_run=8_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1454_episode=90_run=8_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1454_episode=90_run=8_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1468_episode=90_run=9_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1468_episode=90_run=9_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1468_episode=90_run=9_epochs=5000_v3.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1371_episode=90_run=10_epochs=5000.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1371_episode=90_run=10_epochs=5000_v2.p",
    "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1371_episode=90_run=10_epochs=5000_v3.p",
]

space_uncertainty = [
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=0_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=0_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=0_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=1_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=1_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=1_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=2_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=2_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=2_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=3_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=3_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=3_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6290_episode=20_run=4_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6290_episode=20_run=4_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6290_episode=20_run=4_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=5_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=5_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=5_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6283_episode=20_run=6_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6283_episode=20_run=6_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6283_episode=20_run=6_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=7_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=7_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6278_episode=20_run=7_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=8_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=8_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6279_episode=20_run=8_epochs=5000_v3.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=9_epochs=5000.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=9_epochs=5000_v2.p',
    'SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len6300_episode=20_run=9_epochs=5000_v3.p',
]

breakout_uncertainty = [
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1630_episode=90_run=0_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1630_episode=90_run=0_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1630_episode=90_run=0_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1580_episode=90_run=1_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1580_episode=90_run=1_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1580_episode=90_run=1_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1406_episode=90_run=3_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1406_episode=90_run=3_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1406_episode=90_run=3_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1518_episode=90_run=4_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1518_episode=90_run=4_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1518_episode=90_run=4_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1622_episode=90_run=6_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1622_episode=90_run=6_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1622_episode=90_run=6_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1608_episode=90_run=8_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1608_episode=90_run=8_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1608_episode=90_run=8_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode=90_run=9_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode=90_run=9_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode=90_run=9_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1566_episode=90_run=11_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1566_episode=90_run=11_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1566_episode=90_run=11_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1434_episode=90_run=12_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1434_episode=90_run=12_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1434_episode=90_run=12_epochs=5000_v2.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode=90_run=14_epochs=5000.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode=90_run=14_epochs=5000_v3.p',
    'Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1_len1564_episode=90_run=14_epochs=5000_v2.p' ,
]

freeway_uncertainty_7000 = [
    'Results/UncertaintyModels/epoch=9999bsize=7000l=8212_e=499__r=4Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=8172_e=499__r=0Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=8141_e=499__r=9Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=8138_e=499__r=8Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=8028_e=499__r=6Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7992_e=499__r=2Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7969_e=499__r=5Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7968_e=499__r=1Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7784_e=499__r=3Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7776_e=499__r=7Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
]
freeway_uncertainty_3000=[
    'Results/UncertaintyModels/epoch=9999bsize=3000l=8212_e=499__r=4Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=8172_e=499__r=0Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=8141_e=499__r=9Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=8138_e=499__r=8Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=8028_e=499__r=6Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7992_e=499__r=2Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7969_e=499__r=5Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7968_e=499__r=1Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7784_e=499__r=3Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7776_e=499__r=7Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
]
freeway_uncertainty_1000=[
    'Results/UncertaintyModels/epoch=9999bsize=1000l=8212_e=499__r=4Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=8172_e=499__r=0Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=8141_e=499__r=9Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=8138_e=499__r=8Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=8028_e=499__r=6Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7992_e=499__r=2Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7969_e=499__r=5Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7968_e=499__r=1Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7784_e=499__r=3Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7776_e=499__r=7Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7].p',
]

space_uncertainty_7000 = [
    'Results/UncertaintyModels/epoch=9999bsize=7000l=12000_e=39__r=9SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11975_e=39__r=5SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11958_e=39__r=6SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11956_e=39__r=3SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11950_e=39__r=4SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11940_e=39__r=2SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11934_e=39__r=0SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11933_e=39__r=1SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11885_e=39__r=8SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=11873_e=39__r=7SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
]
space_uncertainty_3000 = [
    'Results/UncertaintyModels/epoch=9999bsize=3000l=12000_e=39__r=9SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11975_e=39__r=5SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11958_e=39__r=6SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11956_e=39__r=3SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11950_e=39__r=4SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11940_e=39__r=2SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11934_e=39__r=0SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11933_e=39__r=1SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11885_e=39__r=8SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=11873_e=39__r=7SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
]
space_uncertainty_1000 = [
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11873_e=39__r=7SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11885_e=39__r=8SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11933_e=39__r=1SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11934_e=39__r=0SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11940_e=39__r=2SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11950_e=39__r=4SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11956_e=39__r=3SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11958_e=39__r=6SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=11975_e=39__r=5SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=12000_e=39__r=9SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6].p',
]

breakout_uncertainty_7000 = [
    'Results/UncertaintyModels/epoch=9999bsize=7000l=8114_e=499__r=2Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7940_e=499__r=0Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7768_e=499__r=6Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7726_e=499__r=5Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7614_e=499__r=7Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7544_e=499__r=3Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7426_e=499__r=1Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7420_e=499__r=4Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7142_e=499__r=8Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=7000l=7608_e=499__r=9Breakout_CorruptedStates=[2, 4].p',
]
breakout_uncertainty_3000 = [
    'Results/UncertaintyModels/epoch=9999bsize=3000l=8114_e=499__r=2Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7940_e=499__r=0Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7768_e=499__r=6Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7726_e=499__r=5Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7614_e=499__r=7Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7544_e=499__r=3Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7426_e=499__r=1Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7420_e=499__r=4Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7142_e=499__r=8Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=3000l=7608_e=499__r=9Breakout_CorruptedStates=[2, 4].p',
]
breakout_uncertainty_1000 = [  
    'Results/UncertaintyModels/epoch=9999bsize=1000l=8114_e=499__r=2Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7940_e=499__r=0Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7768_e=499__r=6Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7726_e=499__r=5Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7614_e=499__r=7Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7544_e=499__r=3Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7426_e=499__r=1Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7420_e=499__r=4Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7142_e=499__r=8Breakout_CorruptedStates=[2, 4].p',
    'Results/UncertaintyModels/epoch=9999bsize=1000l=7608_e=499__r=9Breakout_CorruptedStates=[2, 4].p',
]

s_vf_list = [0.01]
s_md_list = [0.1]
model_corruption_list = [0]
model_list = [{'type': 'heter', 'layers_type': ['fc'], 'layers_features': [6], 'action_layer_num': 2}]
experiment_detail = ""

index = 9


# agent
rollout_idea = 5 # None, 1, 5
selection_idea = 1  # None, 1
backpropagate_idea = 1  # None, 1
expansion_idea = 2 # None, 2,
pre_gathered_buffer = None #'l=4954_e=299__r=0_Freeway_SemiOnlineUAMCTS_R=N_E=2_S=N_B=N_AdaptiveTau=10_5000_5000_Run9' # freeway_buffer[index // 3]
# load_vf = ["Results/ValueFunction/Space_Invaders_Run1_64x64_VF.p", 
#            "Results/ValueFunction/freeway_Run1_64x64_VF.p",
#            "Results/ValueFunction/breakout_Run1_64x64_VF.p"][0]
# 3, 7 , 20
# 7, 10, 20
# 7, 10, 20
trained_vf_list = [None]#["Results/ValueFunction/breakout_run"+str(i)+"_E20000_64x64_VF.p" for i in range(30)]
# Uncertainty

# # experiment
num_runs = 1
num_episode = 2000 #spc=300 frw=1500 brk=2000
max_step_each_episode = 300

u_batch_size = 32
minimum_uncertainty_buffer_training = u_batch_size
u_step_size = 0.001
u_layers_type = ['fc', 'fc']
u_layers_features = [128, 128]
u_training = True
u_epoch_training = 5000
u_epoch_training_rate = 5000

#v1 128 128 128 128 - 5k -5k
#v2 128 128 - 5k - 5k
#v3 128 128 - 3k - 5k
#v4 128 64 - 5k - 5k



u_training_steps = [u_epoch_training_rate * i for i in range(num_episode * max_step_each_episode // u_epoch_training_rate)]

# u_pretrained_u_network = freeway_uncertainty_1000[index]
u_pretrained_u_network = None #'/home/farnaz/UAMCTS/PretrainedUncertaintyModels/uf128-5' #"/home/farnaz/UAMCTS/PretrainedUncertaintyModels/UAMCTS(E)_Tau=10_r0_b=4710_e=3000.p"
use_perfect_uncertainty = False

#environment
env_name = "breakout" #freeway, breakout, space_invaders

# c_list = [2 ** -1, 2 ** 0, 2 ** 0.5, 2 ** 1]
c_list = [2 ** 0.5] #space_invaders=2**0, freeway=2**0.5, breakout=2**0.5
                # mcts: space_invaders=2**1, freeway=2**0.5, breakout=2**-1
num_iteration_list = [100] #[10] [100]
simulation_depth_list = [50] #[20] [50]
num_simulation_list = [10] #[10] 
# tau_list = [0.1] #space_invaders=0.1, freeway=0.1, breakout=0.1 0.5
tau_list = [10]
save_uncertainty_buffer = True



test = False
if test:
    result_file_name = "t"
else:
    # result_file_name = "SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=5_E=N_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy_I"+str(index)
    # result_file_name = "SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run4"
    # result_file_name = "Freeway_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run11"
    # result_file_name = "2_Freeway_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run4"
    result_file_name = "3_Breakout_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run7"
    # result_file_name = "2_SpaceInvaders_SemiOnlineUAMCTS_R=5_E=2_S=1_B=1_Tau=10_Run9"
    # result_file_name = "Freeway_MCTS_Run9"
    # result_file_name = "TestTau_Freeway_SemiOnlineUAMCTS_R=N_E=2_S=N_B=N_" + str(tau_list[0])
    # result_file_name = "Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=2_S=N_B=N_TrainedUncertainty_B1000_ParameterStudy_I"+str(index)
    # result_file_name = "TwoWayGridWorld_CorruptedStates=[3]_MCTS_BootstrapDQNValue_Offline_ParameterStudy_run1"

    # result_file_name = "SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1"
    # result_file_name = "Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_PreTrainedDQN_E30000_64x64_ParameterStudy_run1"
    # result_file_name = "Breakout_CorruptedStates=[2, 4]_PreTrainedDQN_E20000_64x64_ParameterStudy_run1"
