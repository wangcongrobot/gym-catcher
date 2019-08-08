# using openai baselines==0.1.5
train the model:
python train_ur5_pick.py --logdir log --num_cpu 16

evaluate the model:
python play_ur5_pick.py log/policy_best.pkl

