import math
from scipy.special import expit
import torch
import itertools
import numpy as np
import random

import torch.nn.functional as F

from LTRModel.utils import torch_dcg_at_k
from LTRModel.ranknet_model import LTRModel_RankNet
from LTRModel.listnet_model import LTRModel_ListNet

class ModelInterface:
    # Maybe change loss_function and optimizer
    def __init__(self, nr_epochs = 1000, list_len=2, sample_size = 10, batch_size = 100, loss_function = "ranknet", function = "", save = True):
        
        self.ltr_net = None
        self.epochs = nr_epochs
        self.list_len = list_len
        
        self.sample_size = sample_size
        self.batch_size = batch_size
        
        self.loss_function = loss_function
        self.function = function
        #self.optimizer = optimizer
        self.save = save
        
    def load(self, path):
        pass
    
    
    def fit(self, X_train, y_train, X_test, y_test):
        # Get at first the possible combinations for one query
        # The number of features --> TODO make changeable
        input_channels = 9 # TODO needs to be changed
        if self.list_len == 2:
            self.ltr_net = LTRModel_RankNet(input_channels)
        elif self.list_len > 2:
            self.ltr_net = LTRModel_ListNet(input_channels)
        else:
            raise Error("Length of list not valid")
        # self.ltr_net.zero_grad() --> Necessary?
        all_losses = []
        optimizer = torch.optim.Adam(self.ltr_net.parameters())
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        loss_function = torch.nn.BCELoss()

        test_len = len(list(X_test.keys()))
        # Do preprocessing stuff and so on here
        for epoch in range(self.epochs):
            counter = 0
            losses = 0
            
            job_numbers = list(X_train.keys())
            
            curr_job_numbers = random.sample(job_numbers, self.batch_size)
           # print(curr_job_numbers)
            for idx,x in enumerate(curr_job_numbers): 
                x_keys = list(X_train[x].keys())
                #print(x_keys)
                
                length = self.sample_size if len(x_keys) > self.sample_size else len(x_keys)
                curr_keys = random.sample(x_keys, length)                

                for indices in itertools.combinations(np.arange(length), self.list_len):
                    if indices[0] == indices[1]:
                        continue
                        
                    keys = [curr_keys[i] for i in indices]
                    curr_x = [X_train[x][key] for key in keys]
                    curr_y = [y_train[key] for key in keys]
                    
                    #if self.has_identical(curr_y):
                    #    continue
                    if self.all_same(curr_y):
                        continue
                    
                    counter += 1
                #sample_length = len(x)
                #comb_comp = math.factorial(sample_length)/(math.factorial(sample_length-self.list_len)*math.factorial(self.list_len))
                #act_comparisons = min(comb_comp, comparisons_per_set)
                    
                    # Todo extent for more than 2 
                    #for idx in indices:
                    
                    pred_result = self.ltr_net(curr_x)
                    #print(pred_result)
                    if self.list_len == 2:
                        if self.loss_function == "ranknet":
                            expected_result = torch.tensor([[expit(curr_y[0]-curr_y[1])]], dtype=torch.float32) # expit is the weird scipy function for sigmoid
                            loss = loss_function(pred_result, expected_result)
                        elif self.loss_function == "frank":
                            expected_result = torch.tensor([[expit(curr_y[0]-curr_y[1])]], dtype=torch.float32) # expit is the weird scipy function for sigmoid
                            loss = 1 - torch.sqrt(expected_result*pred_result)-torch.sqrt((1-expected_result)*(1-pred_result))
                            
                    else:
                        if self.loss_function == "listnet":
                            expected_result = F.softmax(torch.tensor(curr_y).reshape(1,-1).t(), dim=0)
                            pred_result = F.log_softmax(pred_result, dim = 0)
                            loss = -torch.sum(expected_result * pred_result)
                        elif self.loss_function == "softrank":
                            batch_mus  = pred_result
                            batch_pairsub_mus = torch.unsqueeze(batch_mus, dim=2) - torch.unsqueeze(batch_mus, dim=1)
                            # variance w.r.t. s_i - s_j, which is equal to sigma^2_i + sigma^2_j
                            pairsub_vars = 2 * torch.tensor([[1.0]])**2
                            # \Phi(0)$
                            batch_Phi0 = 0.5 * torch.erfc(batch_pairsub_mus / torch.sqrt(2 * pairsub_vars))
                            # remove diagonal entries
                            batch_Phi0_subdiag = torch.triu(batch_Phi0, diagonal=1) + torch.tril(batch_Phi0, diagonal=-1)
                            batch_expt_ranks = torch.sum(batch_Phi0_subdiag, dim=2) + 1.0

                            batch_gains = torch.pow(2.0, torch.tensor(curr_y).reshape(1,-1)) - 1.0
                            batch_dists = 1.0 / torch.log2(batch_expt_ranks + 1.0)  # discount co-efficients
                            batch_idcgs = torch_dcg_at_k(batch_rankings=torch.tensor(curr_y).reshape(1,-1), label_type="Multilabel")
                            batch_dcgs = batch_dists * batch_gains
                            batch_expt_nDCG = torch.sum(batch_dcgs/batch_idcgs, dim=1)
                            loss = - torch.sum(batch_expt_nDCG)
                    optimizer.zero_grad()
                        
                        #print(loss)
                    losses += loss
                    loss.backward()
                    #for name, param in self.ltr_net.named_parameters():
                    #    print(name, torch.isfinite(param.grad).all(), torch.max(abs(param.grad)))
                    optimizer.step()
                    
            
            found = 0
            avg_score_best = 0
            ignored = 0
            best_found = 0
            for x_test in X_test.keys():
                test_data = list(X_test[x_test].values())
                if not test_data:
                    ignored += 1
                    continue
                y_predicted = np.array(self.ltr_net.predict_all(test_data))
                y_true = []
                for y in X_test[x_test].keys():
                    y_true.append(y_test[y])
                #print(y_predicted, y_true)    
                if np.argmax(y_predicted) == np.argmax(np.array(y_true)):
                    found += 1
                avg_score_best += y_true[np.argmax(y_predicted)]
            if counter != 0:    
                curr_loss = losses/counter
            avg = avg_score_best/(test_len-ignored)
            if avg > best_found:
                best_found = avg
                torch.save(self.ltr_net.state_dict(), f"./LTRModel/models/best_model_test_2_epoch_{epoch}_{self.loss_function}.pth")
            print(f"Epoch: {epoch} Loss: {curr_loss} Best found: {found}/{test_len-ignored} Avg. Score Best: {avg}")
            all_losses.append(curr_loss)
            #scheduler.step()
        #if self.save:
        #    torch.save(self.ltr_net.state_dict(), "./LTRModel/models/last_model_test_2_.pth")
    
    def has_identical(self, ys):
        visited_y = []
        for y in ys:
            if y in visited_y:
                return True
            else:
                visited_y.append(y)
        return False
            
    def all_same(self, ys):
        arr = np.array(ys)
        return len(np.unique(arr)) == 1
        
        