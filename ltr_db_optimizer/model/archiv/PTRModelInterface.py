import math
from scipy.special import expit
import torch
import itertools
import numpy as np
import random

import torch.nn.functional as F

from ptranking.ltr_adhoc.pairwise.ranknet import RankNet, RankNetParameter
from LTRModel.ptranknet_model import LTRModel_RankNet

def create_model(LossFunction):
    
    class ModelInterface(LossFunction):
        def initialize(self):
            ...
        
        def __init__(self, epochs = 100, sample_size = 10, batch_size = 100):
            #super().__init__()
            self.epochs = epochs
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.optimizer = None
            self.sigma = 1.0
        
        def fit(self, X_train, y_train, X_test, y_test):
            input_channels = 9
            
            self.net = LTRModel_RankNet(input_channels)
            self.optimizer = torch.optim.Adam(self.net.parameters())
            all_losses = []
            test_len = len(list(X_test.keys()))
            
            for epoch in range(self.epochs):
                counter = 0
                losses = 0
                
                job_numbers = list(X_train.keys())
                curr_job_numbers = random.sample(job_numbers, self.batch_size)
                
                for idx, x in enumerate(curr_job_numbers):
                    x_keys = list(X_train[x].keys())
                    length = self.sample_size if len(x_keys) > self.sample_size else len(x_keys)
                    curr_keys = random.sample(x_keys, length)
                    
                    curr_x = [X_train[x][key] for key in curr_keys]
                    curr_y = torch.Tensor([[y_train[key] for key in curr_keys]])
                    
                    counter += 1
                    pred_results = self.net(curr_x).t()
                    
                    loss = self.custom_loss_function(pred_results, curr_y)
                    losses += loss
                    
                found = 0
                avg_score_best = 0
                ignored = 0
                best_found = 0
                for x_test in X_test.keys():
                    test_data = list(X_test[x_test].values())
                    y_predicted = np.array(self.net.predict_all(test_data))
                    y_true = []
                    for y in X_test[x_test].keys():
                        y_true.append(y_test[y])
                    if np.argmax(y_predicted) == np.argmax(np.array(y_true)):
                        found += 1
                    avg_score_best += y_true[np.argmax(y_predicted)]
                if counter != 0:    
                    curr_loss = losses/counter
                avg = avg_score_best/(test_len-ignored)
                #if avg > best_found:
                #    best_found = avg
                    #torch.save(self.ltr_net.state_dict(), f"./LTRModel/models/best_model_test_2_epoch_{epoch}_{self.loss_function}.pth")
                print(f"Epoch: {epoch} Loss: {curr_loss} Best found: {found}/{test_len-ignored} Avg. Score Best: {avg}")
                all_losses.append(curr_loss)
                    
            
            
    return ModelInterface()
        
        #def __init__(self):
          