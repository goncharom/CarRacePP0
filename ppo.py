import torch
import torch.nn.functional as F
import numpy as np 
from tensorboardX import SummaryWriter
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
from torch.distributions import Beta
from model_and_utils import *
from itertools import count



class PPO():
	def __init__(self, lr, env, stack_size, eval_env, init_ob):
		self.stack_size = stack_size
		self.env = env
		self.eval_env = eval_env
		self.network = actorCritic(self.stack_size)
		self.old_network = actorCritic(self.stack_size)
		self.dic_placeholder = self.network.state_dict()
		self.old_network.load_state_dict(self.dic_placeholder)
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
		self.last_ob = rgb2gris(init_ob)
	def experience(self, steps):

		total_obs = np.zeros((steps, )+self.last_ob.shape+(self.stack_size,))

		total_rewards = np.zeros((steps, 1))
		total_actions = np.zeros((steps, 3))
		total_values = np.zeros((steps+1, 1))
		masks = np.zeros((steps, 1))

		for step in range(steps):
			total_obs[step] = np.roll(total_obs[step], shift=-1, axis=-1)
			total_obs[step, :, :, -1] = self.last_ob

			alpha, beta, values = self.network(torch.from_numpy(total_obs[step]).type(torch.FloatTensor).unsqueeze(0))
			total_values[step] = values.view(-1).detach().numpy()
			
			m = Beta(alpha, beta)
			actions = m.sample()

			total_actions[step] = actions.numpy()
			actions  =  actions.numpy()*np.array([2., 1., 1.]) - np.array([1., 0., 0.])
			actions = actions.reshape((-1)) 
			
			self.last_ob, rews, dones_, _ = self.env.step(actions)
			self.env.render()

			self.last_ob = rgb2gris(self.last_ob)
			dones = np.logical_not(dones_)*1
			total_rewards[step] = rews
			masks[step] = dones
			if dones_:
				self.env.reset()

		temp_ob = np.roll(total_obs[step], shift=-1, axis=-1)
		temp_ob[..., -1] = self.last_ob
		_, _, values = self.network(torch.from_numpy(temp_ob).type(torch.FloatTensor).unsqueeze(0))
		total_values[steps] = values.view(-1).detach().numpy()

		advantage, real_values = gae(total_rewards, masks, total_values)
		advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

		return(total_obs, total_values, total_rewards, total_actions, masks, advantage, real_values)
	def eval(self):
		ob = self.eval_env.reset()
		self.eval_env.render()
		eval_rewards = 0
		for e in count():
			eval_probs, _ = self.network(torch.from_numpy(ob).type(torch.FloatTensor))
			eval_m = Categorical(eval_probs)
			eval_action = eval_m.sample()
			ob, eval_rews, done, _ = self.eval_env.step(eval_action.numpy())
			self.eval_env.render()

			eval_rewards += eval_rews
			if done:
				break
		return(eval_rewards)		



	def update(self, epochs, steps, total_obs, total_actions, advantage, real_values):


		total_obs_ = torch.from_numpy(total_obs).type(torch.FloatTensor)
		advantage_ = torch.from_numpy(advantage).type(torch.FloatTensor)
		real_values_ = torch.from_numpy(real_values).type(torch.FloatTensor)
		total_actions = torch.from_numpy(total_actions).type(torch.FloatTensor)


		for _ in range(epochs):
			inds = np.arange(steps)
			np.random.shuffle(inds)

			for t in range(steps):
				index = inds[t]

				alpha, beta, values_to_backprop = self.network(total_obs_[index].unsqueeze(0))

				m = Beta(alpha, beta)
				action_taken_prob = m.log_prob(total_actions[index]).sum(dim=1, keepdim=True)


				entropy = m.entropy()
				entropy = entropy.sum(dim=1)
				print(entropy)

				alpha, beta, _ = self.old_network(total_obs_[index].unsqueeze(0))
				m_old = Beta(alpha, beta)
				old_action_taken_probs = m_old.log_prob(total_actions[index]).sum(dim=1, keepdim=True)

				ratios = action_taken_prob/(old_action_taken_probs + 1e-5)

				surr1 = ratios * advantage_[index]
				surr2 = torch.clamp(ratios, min=(1.-.1), max=(1.+.1))*advantage_[index]
				policy_loss = -torch.min(surr1, surr2)
				value_loss = ((values_to_backprop-real_values_[index])**2)
				#value_loss = F.smooth_l1_loss(values_to_backprop, real_values_[index])
				total_loss = policy_loss+value_loss-0.01*entropy
				print(total_loss)
				self.optimizer.zero_grad()
				total_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
				self.optimizer.step()


		self.old_network.load_state_dict(self.dic_placeholder)
		self.dic_placeholder = self.network.state_dict()
		return (value_loss)