import torch
import torch.nn as nn
from .Meso import MesoInception4, Meso4

class MesoDeception(nn.Module):

	def __init__(self):
		super(MesoDeception, self).__init__()
		self.model1 = MesoInception4(num_classes=1)
		self.model2 = MesoInception4(num_classes=1)
	
	def forward(self, input):
		real_score = self.model1(input)
		fake_score = self.model2(input)
		
		return torch.cat((real_score, fake_score), axis = 1)