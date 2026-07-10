import math
import torch
import itertools
import numpy as np
import random
import os
import pickle

import torch.nn.functional as F

from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.listnet import ListNet
from ltr_db_optimizer.model.model_structures.comparison_net2 import LTRComparisonNet ##changed

from ltr_db_optimizer.model.model_structures.proposals_by_XL import *


from ltr_db_optimizer.model.metrics import PositionK, HitsAtK, TopKFound, FoundBestK


from ltr_db_optimizer.model.metrics import ndcg

from ltr_db_optimizer.ext.ptranking.data.data_utils import LABEL_TYPE

from datetime import datetime

def ndcg_wrap(y_pred, y_true, ats=None):
    return ndcg(torch.tensor([y_pred]), torch.tensor([y_true]), ats=ats).numpy()


def create_model(LossFunction, **kwargs):
    class ModelInterface(LossFunction):
        def initialize(self):
            ...

        def __init__(self, epochs=251, sample_size=15, batch_size=100, name="", folder="", **kwargs):
            super().__init__(sf_para_dict={"sf_id": 'pointsf', "opt": None, "lr": None}, **kwargs)
            self.epochs = epochs
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.optimizer = None
            self.device = None

            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

            print('CUDA available:', torch.cuda.is_available())
            self.name = name
            self.folder = folder

            # for position k
            self.position_k = PositionK()
            self.found_k = TopKFound(5)  # metric defines one of the top k plans was predicted as best plan
            self.best_k_of_k = FoundBestK(5)  # metric to find the best position of the top k plans


        def fit(self, X_train_vecs, X_train_tree, y_train, X_test_vecs, X_test_trees, y_test, use_presort=False,
                use_scheduler=False, optimizer="adam"):
            input_dim_1 = 10  # 9#10
            input_dim_2 = 6  # 4 #3# changed: input_dim_2 = 4
            model_archi_name = self.name.split("MODEL_")[1].split("_")[0]
            print('model_archi_name: ', model_archi_name)

            if model_archi_name == "HM":
                self.net = LTRComparisonNet(input_dim_1, input_dim_2).to(self.device)
                print('load LTRComparisonNet HM!!!')
            else:
                self.net = eval(model_archi_name)(input_dim_1, input_dim_2).to(self.device)
                print(f'load XL proposed model:{model_archi_name}!!!')
                print('load LTRComparisonNet XL!!!')

            if optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
            elif optimizer == "adagrad":
                self.optimizer = torch.optim.Adagrad(self.net.parameters())
            elif optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)
            elif optimizer == "rmsprop":
                self.optimizer = torch.optim.RMSprop(self.net.parameters())

            if use_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

            test_len = len(list(X_test_trees.keys()))

            if not os.path.exists(self.folder + "/" + self.name):
                os.makedirs(self.folder + "/" + self.name)


            best_ndcg = 0

            min_valid_loss = 1e8

            result_dict = {}

            print('start training: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            training_loss = []
            all_valid_loss = []

            for epoch in range(self.epochs):
                print('epoch: ', epoch)
                counter = 0
                train_loss_epoch = 0

                job_numbers = list(X_train_vecs.keys())
                total_number_of_jobs = len(job_numbers)

                self.net.train()
                num_batches = len(job_numbers)//self.batch_size if len(job_numbers)%self.batch_size ==0 else (len(job_numbers)//self.batch_size + 1)
                print('number of batches: ', num_batches)

                for batch_i in range(num_batches):
                    curr_job_numbers = job_numbers[batch_i*self.batch_size:(batch_i+1)*self.batch_size]

                    for idx, x in enumerate(curr_job_numbers):
                        curr_keys = list(X_train_vecs[x].keys())
                        length = self.sample_size if len(curr_keys) > self.sample_size else len(curr_keys)
                        # random.shuffle(curr_keys)
                        # curr_keys = random.sample(curr_keys, length)

                        curr_x = [X_train_tree[x][key] for key in curr_keys]
                        if len(curr_x) == 1:
                            continue

                        # print('original type of curr_x:', type(curr_x))
                        # curr_x = torch.Tensor(curr_x).to(self.device)

                        # print('the first 5 training query vectors: ', [np.array(X_train_vecs[x][key]) for key in curr_keys][:5])
                        curr_x_vec = [X_train_vecs[x][key] for key in curr_keys][0]
                        # print('original type of curr_x_vec:', type(curr_x_vec))
                        curr_x_vec = torch.Tensor(curr_x_vec).to(self.device)
                        print('batch index, job_index: ', batch_i, idx)
                        print('curr_x shp: ',  len(list(curr_x)), list(map(lambda x:len(x), curr_x)))
                        print('curr_x_vec shp: ', curr_x_vec.shape)


                        curr_y = torch.Tensor([[y_train[key] for key in curr_keys]]).to(self.device)

                        if torch.count_nonzero(curr_y) == 0:
                            continue

                        counter += 1
                        pred_results = self.net(curr_x_vec, curr_x).t()

                        if use_presort:
                            curr_y, batch_ideal_desc_inds = torch.sort(curr_y, dim=1, descending=True)
                            pred_results = torch.gather(pred_results, dim=1, index=batch_ideal_desc_inds)

                        print('y predict: ', pred_results, pred_results.shape)
                        print('######')
                        print('y true: ', curr_y, curr_y.shape)
                        print('######')


                        loss = self.custom_loss_function(pred_results, curr_y, label_type=LABEL_TYPE.MultiLabel, presort=use_presort)
                        print('loss: ', loss.item())

                        print('y train true and pred: ', curr_y[0], pred_results[0].detach().cpu().numpy())
                        train_ndcg_value = ndcg_wrap(pred_results[0].detach().cpu().numpy(), curr_y[0].detach().cpu().numpy())
                        print('train ndcg: ', train_ndcg_value)

                        print('######')

                        train_loss_epoch += loss.item()

                    with torch.no_grad():

                        ignored = 0
                        print('current test jobs: ', X_test_vecs.keys())

                        ndcg_all_test = []

                        valid_losses = []

                        for x_test in X_test_vecs.keys():
                            test_data_vec = list(X_test_vecs[x_test].values())[0]
                            test_data_tree = list(X_test_trees[x_test].values())

                            if not test_data_vec or len(test_data_tree) == 1:
                                ignored += 1
                                continue
                            # print('original type of test_data_vec:', type(test_data_vec))
                            test_data_vec = torch.Tensor(test_data_vec).to(self.device)
                            # test_data_tree = torch.Tensor(test_data_tree).to(self.device)

                            y_predicted = self.net.predict_all(test_data_vec, test_data_tree).t()
                            y_true = []
                            for y in X_test_vecs[x_test].keys():
                                y_true.append(float(y_test[y]))


                            print('validate y_predicted', y_predicted.detach().cpu().numpy())
                            print('validate y_true: ', y_true)

                            valid_loss = self.custom_loss_function(y_predicted, torch.tensor([y_true]).to(self.device), train=False, label_type=LABEL_TYPE.MultiLabel)
                            print('validate loss: ', valid_loss.item())

                            valid_losses.append(valid_loss.item())

                            ndcg_value = ndcg_wrap(y_predicted[0].detach().cpu().numpy(), y_true)

                            print('validate ndcg', ndcg_value)
                            ndcg_all_test.append(ndcg_value[0][0])

                        avg_ndcg = np.mean(ndcg_all_test)
                        print('average ndcg: ', avg_ndcg)
                        avg_valid_loss = np.mean(valid_losses)
                        print('average valid loss: ', avg_valid_loss)

                        all_valid_loss.append((epoch, batch_i, avg_valid_loss))


                        if avg_ndcg > best_ndcg:
                            best_ndcg = avg_ndcg
                            torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/best_avg_ndcg.pth")

                        if avg_valid_loss < min_valid_loss:
                            min_valid_loss = avg_valid_loss
                            torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/min_avg_valid_loss.pth")

                    if use_scheduler:
                        scheduler.step()
                        print('current lr:', scheduler.get_lr())

                avg_train_loss_epoch = round(train_loss_epoch / total_number_of_jobs, 2)
                training_loss.append((epoch, avg_train_loss_epoch))

            result_dict['train_loss'] = training_loss
            result_dict['valid_loss'] = all_valid_loss
            with open(self.folder + "/" + self.name + "/info.pickle", "wb") as f:
                pickle.dump(result_dict, f)

            print('finish training: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return ModelInterface(**kwargs)
