
import torch.nn


from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from ltr_db_optimizer.ext.TreeConvolution.util_feature import prepare_trees

from ltr_db_optimizer.allrank.models.transformer import  attention



class LTRankNet0(torch.nn.Module):
    """
    use attention mechanism as weights

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 256),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )



        self.comparison_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 256),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU()
        )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(16 + 32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, query_enc, sample_trees, plan_num=10):
        assert len(sample_trees) > 1
        # print('sample trees: ', sample_trees[:5])
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        # print('trees: ', trees[:5])
        comparisons = self.comparison_net(trees)
        ### 3D tensor and attention

        objects_shp = objects.shape
        objects = torch.reshape(objects, (-1, plan_num, objects_shp[-1]))
        comparisons_shp = comparisons.shape
        comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        print('new comparison, objects: ', comparisons.shape, objects.shape)

        # print('comparisons: ', comparisons[:5])
        comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        # print('forward attention results: ', comparison_sums[:5], attention_weights[:5])

        with_query_enc = torch.cat((objects, comparison_sums), 2)
        print('NN with_query_enc: ', with_query_enc.shape, with_query_enc[:5])
        output = self.output_net(with_query_enc)

        # print('NN output: ', output.shape, output[:5])  # torch.Size([62, 10, 1])
        output = torch.squeeze(output, -1)

        return output

    def predict_all(self, query_enc, sample_trees, plan_num=5):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        ### 3D tensor and attention

        objects_shp = objects.shape
        objects = torch.reshape(objects, (-1, plan_num, objects_shp[-1]))
        comparisons_shp = comparisons.shape
        comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        print('new comparison, objects: ', comparisons.shape, objects.shape)

        comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 2)
        print('NN with_query_enc: ', with_query_enc.shape)
        output = self.output_net(with_query_enc)

        print('NN output: ', output.shape)  # torch.Size([62, 10, 1])
        output = torch.squeeze(output, -1)

        return output


    def online_predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)#*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))

        # print('query_enc.is_cuda', query_enc.is_cuda)

        query = self.query_net(query_enc)
        # print('hahahyeye')

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)

        flat_trees, indexes = trees
        # print('Online NN flat_trees shp: ', flat_trees.shape)
        # print('Online NN indexes: ', indexes.shape)

        objects = self.object_net(trees)#.repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        # print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])

        comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        # print('online attention results: ', comparison_sums[:5], attention_weights[:5])

        with_query_enc = torch.cat((objects, comparison_sums),1)

        output = self.output_net(with_query_enc)
        # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])

        return output


