import torch.nn 
from LTRModel.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from LTRModel.feature_extraction import get_right_child, get_left_child, get_features
from LTRModel.TreeConvolution.util import prepare_trees

# This code snippet was mostly extracted, with some renaming from: 
# https://github.com/learnedsystems/BaoForPostgreSQL/blob/master/bao_server/net.py
class LTRModel_ListNet(torch.nn.Module):
    def __init__(self, input_channels):
        super(LTRModel_ListNet, self).__init__() # Don't know
        self.input_channels = input_channels
        
        # Maybe change this model, but let me try it with this first
        self.ltr_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_channels, 256), # What does this number mean?
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
            torch.nn.Linear(32,1),
            torch.nn.Sigmoid()
        )
        
        self.output_sigmoid = torch.nn.Sigmoid()

    def get_input_channels(self):
        return self.input_channels
    
    def forward(self, samples):
        trees = prepare_trees(samples, get_features, get_left_child, get_right_child)
        
        results = self.ltr_net(trees)
        #result_all = self.output_sigmoid(result_1 - result_2)
        return results
        
    def predict(self, sample):
        result = self.ltr_net(sample)
        return result
    
    def predict_and_get_best(self, samples):
        best_result = None
        best_score = None
        
        for sample in samples:
            result = self.ltr_net(sample)
            if not best_score or result > best_score:
                best_score = result
                best_result = sample
                
        return sample
    
    def predict_all(self, samples):
        y_result = []
        for sample in samples:
            tree = prepare_trees([sample], get_features, get_left_child, get_right_child)
            y_result.append(self.ltr_net(tree).detach().numpy()[0,0])
        return y_result
    
    
    def predict_and_sort(self, samples):
        # TODO
        pass
    
        
                              
