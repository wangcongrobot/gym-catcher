# using openai baselines==0.1.5
train the model:
python train_ur5_pick.py --logdir log --num_cpu 16

evaluate the model:
python play_ur5_pick.py log/policy_best.pkl

# training result

train_success_rate:
![train_success_rate](https://github.com/wangcongrobot/gym-catcher/blob/master/gym_catcher/images/train_success_rate.svg)

test_success_rate:
![test_success_rate](https://github.com/wangcongrobot/gym-catcher/blob/master/gym_catcher/images/test_success_rate.svg)