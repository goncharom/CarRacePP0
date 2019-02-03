import torch
import torch.nn.functional as F
import numpy as np 
from tensorboardX import SummaryWriter
import torch.nn as nn
import gym
import matplotlib.pyplot as plt



total_rewards_to_plot = []
total_updates = []
total_means = []
total_value_loss = []
total_value_loss_means = []
class actorCritic(nn.Module):
	def __init__(self, stack_size):
		super(actorCritic, self).__init__()

		self.convolution1 = nn.Conv2d(stack_size, 8, kernel_size=4, stride=2)#47
		self.convolution2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)#23
		self.convolution3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)#11
		self.convolution4 = nn.Conv2d(32, 64, kernel_size=3, stride=2)#5
		self.fc1 = nn.Linear(320*5, 512)
		self.alpha = nn.Linear(512, 3)
		self.beta = nn.Linear(512, 3)
		self.critic = nn.Linear(512, 1)


	def forward(self, inputs):

		inputs = inputs.view(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])

		x = F.relu(self.convolution1(inputs))
		x = F.relu(self.convolution2(x))
		x = F.relu(self.convolution3(x))
		x = F.relu(self.convolution4(x))
		x = x.view(-1, 320*5)
		x = F.relu(self.fc1(x))
		alph = F.softplus(self.alpha(x))+1
		bet = F.softplus(self.beta(x))+1
		values = self.critic(x)


		return alph, bet, values


def gae (rewards, masks, values):

	gamma = 0.99
	lambd = 0.95

	T, W = rewards.shape
	real_values = np.zeros((T, W))
	advantages = np.zeros((T, W))

	adv_t = 0
	for t in reversed(range(T)):

		delta = rewards[t]*5e-2 + values[t+1] * gamma*masks[t] - values[t]

		adv_t = delta + adv_t*gamma*lambd*masks[t]

		advantages[t] = adv_t

	real_values = values[:T] + advantages

	return advantages, real_values

def rgb2gris(x):

	x = np.dot(x[..., :], [0.299, 0.587, 0.114])
	x = x / 128. - 1.
	return x.reshape((96, -1))
def make_env(rank, env_id):
	def env_fn():
		env = gym.make(env_id)
		env.seed(1+rank)
		env.env.viewer.window.dispatch_events()
		return env
	return env_fn
def plotRewards(rewards, updates_no):
	total_rewards_to_plot.append(rewards)
	total_updates.append(updates_no)
	total_means.append(np.mean(total_rewards_to_plot))
	plt.figure(2)
	plt.clf()
	plt.plot(rewards)
	plt.plot(total_updates, total_rewards_to_plot)
	plt.plot(total_updates, total_means)
	plt.pause(0.001)
def plotValueLoss(valuesLoss):
	total_value_loss.append(float(valuesLoss))
	total_value_loss_means.append(np.mean(total_value_loss))
	plt.figure(1)
	plt.clf()
	plt.plot(total_value_loss_means)
	plt.pause(0.001)