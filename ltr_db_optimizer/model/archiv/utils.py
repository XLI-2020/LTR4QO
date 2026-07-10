import torch
import numpy as np

def torch_dcg_at_k(batch_rankings, cutoff=None, label_type="Multilabel", device='cpu'):
        '''
        ICML-nDCG, which places stronger emphasis on retrieving relevant documents
        :param batch_rankings: [batch_size, ranking_size] rankings of labels (either standard or predicted by a system)
        :param cutoff: the cutoff position
        :param label_type: either the case of multi-level relevance or the case of listwise int-value, e.g., MQ2007-list
        :return: [batch_size, 1] cumulative gains for each rank position
        '''
        if cutoff is None: # using whole list
            cutoff = batch_rankings.size(1)

        if "Multilabel" == label_type:    #the common case with multi-level labels
            batch_numerators = torch.pow(2.0, batch_rankings[:, 0:cutoff]) - 1.0
        #elif LABEL_TYPE.Permutation == label_type: # the case like listwise ltr_adhoc, where the relevance is labeled as (n-rank_position)
        #    batch_numerators = batch_rankings[:, 0:cutoff]
        else:
            raise NotImplementedError
        # no expanding should also be OK due to the default broadcasting
        batch_discounts = torch.log2(torch.arange(cutoff, dtype=torch.float, device=device).expand_as(batch_numerators) + 2.0)
        batch_dcg_at_k = torch.sum(batch_numerators/batch_discounts, dim=1, keepdim=True)
        return batch_dcg_at_k