import torch.nn 
# delete LTRModel. again if needed
from LTRModel.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from LTRModel.feature_extraction_with_query_enc import get_right_child, get_left_child, get_features
from LTRModel.TreeConvolution.util_query_enc import prepare_trees

# This code snippet was mostly extracted, with some renaming from: 
# https://github.com/learnedsystems/BaoForPostgreSQL/blob/master/bao_server/net.py
class LTRModel_RankNet(torch.nn.Module):
    def __init__(self, in_channels_1, in_channels_2):
        super(LTRModel_RankNet, self).__init__() 
        self.input_channels_1 = in_channels_1
        self.input_channels_2 = in_channels_2
        
        self.first_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_channels_1, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,16),
            torch.nn.LeakyReLU() # Is this correct?
        )
        # Maybe change this model, but let me try it with this first
        self.ltr_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_channels_2+16, 256), # What does this number mean?
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()), # TODO MIGHT CHANGE
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,1)#,
            #torch.nn.Sigmoid()
        )
        
        self.output_sigmoid = torch.nn.Sigmoid()

    #def get_input_channels(self):
    #    return self.input_channels
    
    def forward(self,samples_vec, samples_tree):
        assert len(samples_vec) == 2 and len(samples_tree) == 2
        query_enc_1 = self.first_net(torch.Tensor(samples_vec[0]))
        query_enc_2 = self.first_net(torch.Tensor(samples_vec[1]))
        
        tree_1 = prepare_trees([samples_tree[0]], get_features, get_left_child, get_right_child, [query_enc_1])
        tree_2 = prepare_trees([samples_tree[1]], get_features, get_left_child, get_right_child, [query_enc_2])
        
        result_1 = self.ltr_net(tree_1)
        result_2 = self.ltr_net(tree_2)
        
        result_all = self.output_sigmoid(result_1 - result_2)
        return result_all
        
    def predict(self, sample):
        tree = prepare_trees([sample], get_features, get_left_child, get_right_child)
        result = self.ltr_net(tree)
        return result
    
    def predict_and_get_best(self, samples):
        best_result = None
        best_score = None
        
        for sample in samples:
            result = self.predict(sample)
            if not best_score or result > best_score:
                best_score = result
                best_result = sample
                
        return sample
    
    def predict_all(self, samples_vec, samples_tree):
        y_result = []
        for idx,sample in enumerate(samples_vec):
            query_enc = self.first_net(torch.Tensor(sample))
            tree = prepare_trees([samples_tree[idx]], get_features, get_left_child, get_right_child, [query_enc])
            y_result.append(self.ltr_net(tree).detach().numpy()[0,0])
        return y_result
    
    def predict_and_sort(self, samples):
        # TODO
        pass
    
        
                              
