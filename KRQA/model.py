import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator 
entity_dict = {}
relation_dict =  {}
class TransE(nn.Module):
    def __init__(self, entity_num, entity_list, relation_num, relation_list, dim, margin, norm, C):
        super(TransE, self).__init__()
        """ Entity_num is the number of entities
        Relation_num is the number of relations
        dim is the embeded dimension
        margin is the loss function parameters
        norm is the parameters for loss  """
        self.entity_list = entity_list
        self.relation_list = relation_list
        self.entity_num =  entity_num
        self.relation_num = relation_num
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C

        #Embedding network
        self.ent_embeddings = nn.Embedding(self.entity_num, self.dim, max_norm=6/np.sqrt(self.dim), norm_type=self.norm)
        self.rel_embeddings = nn.Embedding(self.relation_num, self.dim, max_norm=6/np.sqrt(self.dim), norm_type=self.norm)
        if torch.cuda.is_available():
            self.ent_embeddings = self.ent_embeddings.cuda()
            self.rel_embeddings = self.rel_embeddings.cuda()

        self.init_embedding(self.entity_num, self.ent_embeddings)
        self.init_embedding(self.relation_num, self.rel_embeddings)
        
    def init_embedding(self, num,  embedding):
        #self.ent_embeddings.weight =  self.ent_embeddings.weight * torch.tensor(2)
        for i in range(num):
            with torch.no_grad():
                embedding.weight[i] = torch.tensor( embedding.weight[i]/(torch.norm(embedding.weight[i])))
    def train(self, data, batch_size = 4096, epoch = 100):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for i in range(epoch):
            self.init_embedding(self.entity_num, self.ent_embeddings)

            batch_samples = random.sample(data, batch_size)
            Tbatch = []
            for sample in batch_samples:
                corrupted_sample = copy.deepcopy(sample)
                if len(corrupted_sample) != 3:
                    continue
                pr = np.random.random(1)[0]
                if pr > 0.5:
                    # change the head entity
                    s = random.sample(self.entity_list, 1)[0]
                    if (len(s) == 1):
                        corrupted_sample[0] = s[0]
                    while corrupted_sample[0] == sample[0]:
                        s = random.sample(self.entity_list, 1)[0]
                        if (len(s) == 1):
                            corrupted_sample[0] = s[0]
                else:
                    # change the tail entity
                    s = random.sample(self.entity_list, 1)[0]
                    if (len(s) == 1):
                        corrupted_sample[2] = s[0]
                    while corrupted_sample[2] == sample[2]:
                        s = random.sample(self.entity_list, 1)[0]
                        if (len(s) == 1):
                            corrupted_sample[2] = s[0]
                if (sample, corrupted_sample) not in Tbatch:
                    Tbatch.append((sample, corrupted_sample))

            loss = self.calculate_loss(Tbatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i, loss.cpu().detach().numpy())
        return 1

    def calculate_loss(self, Tbatch):
        loss = 0
        for i in Tbatch:
            True_Triple = i[0]
            False_Triple = i[1]
            #print(True_Triple[1])
            #print(self.relation_list[0])
            h1 = self.ent_embeddings(torch.tensor(self.entity_list.index([True_Triple[0]])).cuda())
            r1 = self.rel_embeddings(torch.tensor(self.relation_list.index(True_Triple[1])).cuda())
            t1 = self.ent_embeddings(torch.tensor(self.entity_list.index([True_Triple[2]])).cuda())
            
            h2 = self.ent_embeddings(torch.tensor(self.entity_list.index([False_Triple[0]])).cuda())
            r2 = self.rel_embeddings(torch.tensor(self.relation_list.index(False_Triple[1])).cuda())
            t2 = self.ent_embeddings(torch.tensor(self.entity_list.index([False_Triple[2]])).cuda())
            loss1 = (self.margin + torch.norm(h1+r1-t1) - torch.norm(h2+r2-t2))/4096
            if loss1 > 0:
                loss = loss + loss1

        return loss


