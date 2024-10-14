# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np


####==================================================================####
####==============  NEURAL NETWORK IMPLEMENTATION BELOW ==============####
####==================================================================####
class net(nn.Module):
	def __init__(self, 
			  num_features,
			  params, 
			  verbose=False):
		super(net, self).__init__()
		self.model_params = {
			"batch_size": 4096,
			"num_layers": 4,
			"epochs": 10,
			"layer_size": 100,
			"pruning": 1.0,
			"target_weight": 11.5,
			"rate": 0.001,
			"reg": 0.001,
			"lr_decay": 0.98,
			"dropout": 0.3
			}
		self.verbose = verbose
		for param, value in params.items():
			if param in self.model_params.keys():
				self.model_params[param] = value
		self.target_accuracy = 0
		self.nontarget_accuracy = 0
		self.performance = 0
		#print(cols)
	
		current_size = num_features
		for number in np.arange(self.model_params["num_layers"]):
			next_size = int(current_size * self.model_params["pruning"]) if self.model_params["pruning"] < 1.0 else self.model_params["layer_size"]
			setattr(self, "batch{}".format(number), nn.BatchNorm1d(current_size, eps=1e-5, momentum=0.1, track_running_stats = True))
			setattr(self, "linear{}".format(number), nn.Linear(current_size, next_size if number != np.arange(self.model_params["num_layers"])[-1] else 1))
			setattr(self, "relu{}".format(number), nn.GELU())
			current_size = next_size
		
		self.dropout = nn.Dropout(p=self.model_params["dropout"])
		
		#self.softmax = nn.Softmax(dim=1)
		#self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1, target_weight]).float())
		self.optimizer = torch.optim.Adam(self.parameters(), lr = self.model_params["rate"], weight_decay = self.model_params["reg"])
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.model_params["lr_decay"])
		self.sigmoid = nn.Sigmoid()
		self.loss_func = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([self.model_params["target_weight"]]).float())
		self.loss_history = []
		self.validation_loss = []
		
	
	def forward(self, data, targets, test=False):
		x = data
		for i in np.arange(self.model_params["num_layers"]):
			x = getattr(self, "batch"+str(i))(x)
			if not test:
				x = self.dropout(x)
			x = getattr(self, "relu"+str(i))(x)
			x = getattr(self, "linear"+str(i))(x)
		if test:
			return x, 0
		loss = self.loss_func(x[:, 0], targets.float())
		return x, loss

	def train(self, data, targets, validation_subset, validation_targets):
		training_set = torch.tensor(data.values, requires_grad = True).float()
		training_targets_set = torch.tensor(targets.values, requires_grad = True).long()
		
		training_batches = torch.split(training_set, self.model_params["batch_size"])
		target_batches = torch.split(training_targets_set, self.model_params["batch_size"])
		
		
		for epoch in range(self.model_params["epochs"]):
			if self.verbose: print("====== Epoch {} ======".format(epoch))

			for set_num, training_batch, target_batch in zip(range(len(training_batches)), training_batches, target_batches):
				
				scores, loss = self.forward(training_batch, target_batch)
				self.loss_history.append(loss)
				with torch.no_grad():
					self.validation_loss.append(self.forward(validation_subset, validation_targets)[1])
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
			self.scheduler.step()
			
				
		with torch.no_grad():
			scores = self.sigmoid(scores)
			scores[:] = (scores > 0.5).int()

			correct_targets = sum(x > 0.5 and y == 1 for x, y in zip(scores[:, 0], target_batch))
			total_targets = sum(y==1 for y in target_batch)
			target_accuracy = correct_targets/total_targets
			
			correct_nontargets = sum(x < 0.5 and y == 0 for x, y in zip(scores[:, 0], target_batch))
			total_nontargets = sum(y==0 for y in target_batch)
			nontarget_accuracy = correct_nontargets/total_nontargets
			
			performance = target_accuracy + nontarget_accuracy
			
			if performance > self.performance:
				self.performance = performance
				self.target_accuracy = target_accuracy
				self.nontarget_accuracy = nontarget_accuracy
			
			if self.verbose: 
				print("Loss = {:.3f}".format(self.loss))
				print("Target percentage: {:.2f}%".format(100*self.target_accuracy))
				print("Nontarget percentage: {:.2f}%".format(100*self.nontarget_accuracy))
		if self.verbose: print("=======================================")
			
	
	def get_accuracy(self, data, targets = None, test=False):
		with torch.no_grad():
			scores, loss = self.forward(data, targets, test=test)
			#scores = torch.divide(torch.absolute(scores), torch.sum(torch.absolute(scores), axis=1, keepdim=True))
			scores = self.sigmoid(scores)
			if targets == None:
				return scores
			scores[:] = (scores > 0.5).int()
		return scores

	def print_scores(self, test_data):
		self.get_accuracy()
		return
		pass