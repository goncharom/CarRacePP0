import torch
import torch.nn.functional as F
import numpy as np 
from tensorboardX import SummaryWriter
import torch.nn as nn
import gym
from torch.distributions import Categorical
from model_and_utils import *
from ppo import *

#Network parameters

steps = 10000
batch_size = 200
epochs = 3
plot_points = 5 #number of updates between episode reward plot
lr = 2.5e-4
stack_size = 3

total_rewards_to_plot = []
total_updates = []
total_means = []


env = gym.make('CarRacing-v0')

init_ob = env.reset()
env.env.viewer.window.dispatch_events()
eval_env = gym.make('CarRacing-v0')

algo = PPO(lr, env, stack_size, eval_env, init_ob)
iterations = steps/batch_size
for t in range(int(iterations)):
	total_obs, total_values, total_rewards, total_actions, masks, advantage, real_values = algo.experience(batch_size)

	valueLoss = algo.update(epochs, batch_size, total_obs, total_actions, advantage, real_values)
	#plotValueLoss(valueLoss)
	#if (t%plot_points == 0) and (t != 0):
		#rewards_to_plot = algo.eval()
		#plotRewards(rewards_to_plot, t)
	print(t/iterations)


