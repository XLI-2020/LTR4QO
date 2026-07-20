import copy

import torch.nn
import torch.nn as nn


from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from ltr_db_optimizer.ext.TreeConvolution.util_feature import prepare_trees, prepare_trees_plans_only

from ltr_db_optimizer.allrank.models.transformer import make_transformer, MultiHeadedAttention, attention

Attention = True

Transformer = True


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


    # def feature_extract(self, query_enc, sample_trees):
    #     assert len(sample_trees) > 1
    #     query_enc = torch.Tensor(query_enc)#*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
    #     query = self.query_net(query_enc)
    #     print('hahahyeye')
    #     trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
    #
    #     flat_trees, indexes = trees
    #     print('Online NN flat_trees shp: ', flat_trees.shape)
    #     print('Online NN indexes: ', indexes.shape)
    #
    #     objects = self.object_net(trees)#.repeat(1,len(sample_trees))
    #     comparisons = self.comparison_net(trees)
    #
    #     print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])
    #
    #     comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)
    #
    #     print('online attention results: ', comparison_sums[:5], attention_weights[:5])
    #
    #
    #     with_query_enc = torch.cat((objects, comparison_sums),1)
    #
    #     print('extracted features: ', with_query_enc)
    #
    #     # output = self.output_net(with_query_enc)
    #     # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])
    #
    #     return with_query_enc



class LTRankNet0A(torch.nn.Module):
    """
    with context network

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0A, self).__init__()
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
            torch.nn.LeakyReLU()
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
            torch.nn.Linear(32, 64),
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
        # comparisons_shp = comparisons.shape
        # comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        print('new comparison, objects: ', comparisons.shape, objects.shape)

        # print('comparisons: ', comparisons[:5])
        # comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        # print('forward attention results: ', comparison_sums[:5], attention_weights[:5])

        # with_query_enc = torch.cat((objects, comparison_sums), 2)

        with_query_enc = copy.copy(objects)
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
        # comparisons_shp = comparisons.shape
        # comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        # print('new comparison, objects: ', comparisons.shape, objects.shape)
        #
        # comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        # with_query_enc = torch.cat((objects, comparison_sums), 2)
        with_query_enc = copy.copy(objects)
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

        objects = self.object_net(trees) #.repeat(1,len(sample_trees))
        # comparisons = self.comparison_net(trees)

        # print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])

        # comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        # print('online attention results: ', comparison_sums[:5], attention_weights[:5])

        # with_query_enc = torch.cat((objects, comparison_sums),1)
        with_query_enc = copy.copy(objects)

        output = self.output_net(with_query_enc)
        # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])

        return output


    # def feature_extract(self, query_enc, sample_trees):
    #     assert len(sample_trees) > 1
    #     query_enc = torch.Tensor(query_enc)#*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
    #     query = self.query_net(query_enc)
    #     print('hahahyeye')
    #     trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
    #
    #     flat_trees, indexes = trees
    #     print('Online NN flat_trees shp: ', flat_trees.shape)
    #     print('Online NN indexes: ', indexes.shape)
    #
    #     objects = self.object_net(trees)#.repeat(1,len(sample_trees))
    #     comparisons = self.comparison_net(trees)
    #
    #     print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])
    #
    #     comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)
    #
    #     print('online attention results: ', comparison_sums[:5], attention_weights[:5])
    #
    #
    #     with_query_enc = torch.cat((objects, comparison_sums),1)
    #
    #     print('extracted features: ', with_query_enc)
    #
    #     # output = self.output_net(with_query_enc)
    #     # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])
    #
    #     return with_query_enc


class LTRankNet0B(torch.nn.Module):
    """
 replace attention with sum-average aggreation
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0B, self).__init__()
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
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        # print('NN query_enc shp: ', query_enc.shape)
        query = self.query_net(query_enc)
        # print('NN query shp: ', query.shape)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)

        flat_trees, indexes = trees
        print('NN flat_trees shp: ', flat_trees.shape)
        print('NN indexes: ', indexes.shape)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        # print('NN objects: ', objects.shape)
        comparisons = self.comparison_net(trees)
        # print('NN comparisons: ', comparisons.shape)

        ### reshape into batch_size x number_of_plan_per_query x feature_size
        comparisons_shp = comparisons.shape
        comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        objects_shp = objects.shape
        objects = torch.reshape(objects, (-1, plan_num, objects_shp[-1]))

        print('new comparison, objects: ', comparisons.shape, objects.shape)

        ###reshape into batch_size x number_of_plan_per_query x feature_size

        context = torch.sum(comparisons, 1, keepdim=True)
        # print('context: ', context.shape, context[:3]) # torch.Size([62, 1, 16])
        comparison_sums = context - comparisons
        print('comparison_sums ', comparison_sums.shape)

        comparison_sums = comparison_sums / (comparison_sums.shape[1])
        with_query_enc = torch.cat((objects, comparison_sums), 2)
        # print('NN with_query_enc: ', with_query_enc.shape, with_query_enc[:5])
        output = self.output_net(with_query_enc)
        # print('NN output: ', output.shape, output[:5])  # torch.Size([62, 10, 1])

        output = torch.squeeze(output, -1)

        return output

    def predict_all(self, query_enc, sample_trees, plan_num=10):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        ### reshape into batch_size x number_of_plan_per_query x feature_size
        comparisons_shp = comparisons.shape
        print('predict old comparison, objects shp: ', comparisons.shape, objects.shape)
        comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        objects_shp = objects.shape
        objects = torch.reshape(objects, (-1, plan_num, objects_shp[-1]))

        print('predict new comparison, objects: ', comparisons.shape, objects.shape)

        print('predict new comparison sum ', torch.sum(comparisons, 1).shape, torch.sum(comparisons, 1)[:3])

        ###reshape into batch_size x number_of_plan_per_query x feature_size
        context = torch.sum(comparisons, 1, keepdim=True)

        comparison_sums = context - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[1])
        with_query_enc = torch.cat((objects, comparison_sums), 2)

        output = self.output_net(with_query_enc)
        output = torch.squeeze(output, -1)
        return output

    def online_predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # print('start prepare_trees')
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)

        flat_trees, indexes = trees
        # print('Online NN flat_trees shp: ', flat_trees.shape)
        # print('Online NN indexes: ', indexes.shape)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        # print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])

        context = torch.sum(comparisons, 0, keepdim=True)  #
        # print('predict context shp: ', context.shape) # torch.Size([1, 16])

        comparison_sums = context - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('predict new comparison_sums shp: ', comparison_sums.shape) #torch.Size([240, 16])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        output = self.output_net(with_query_enc)
        # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])

        return output

class LTRankNet0B1(torch.nn.Module):
    """
 replace attention with sum-average aggreation
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0B1, self).__init__()
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
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        # print('NN query_enc shp: ', query_enc.shape)
        query = self.query_net(query_enc)
        # print('NN query shp: ', query.shape)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)

        flat_trees, indexes = trees
        print('NN flat_trees shp: ', flat_trees.shape)
        print('NN indexes: ', indexes.shape)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        # print('NN objects: ', objects.shape)
        comparisons = self.comparison_net(trees)
        # print('NN comparisons: ', comparisons.shape)

        ### reshape into batch_size x number_of_plan_per_query x feature_size
        comparisons_shp = comparisons.shape
        comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        objects_shp = objects.shape
        objects = torch.reshape(objects, (-1, plan_num, objects_shp[-1]))

        print('new comparison, objects: ', comparisons.shape, objects.shape)

        ###reshape into batch_size x number_of_plan_per_query x feature_size

        context = torch.sum(comparisons, 1, keepdim=True)
        # print('context: ', context.shape, context[:3]) # torch.Size([62, 1, 16])

        comparison_sums = context.repeat(1, plan_num, 1)


        # comparison_sums = context - comparisons
        print('comparison_sums ', comparison_sums.shape)

        comparison_sums = comparison_sums / (comparison_sums.shape[1])
        with_query_enc = torch.cat((objects, comparison_sums), 2)
        # print('NN with_query_enc: ', with_query_enc.shape, with_query_enc[:5])
        output = self.output_net(with_query_enc)
        # print('NN output: ', output.shape, output[:5])  # torch.Size([62, 10, 1])

        output = torch.squeeze(output, -1)

        return output

    def predict_all(self, query_enc, sample_trees, plan_num=10):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        ### reshape into batch_size x number_of_plan_per_query x feature_size
        comparisons_shp = comparisons.shape
        print('predict old comparison, objects shp: ', comparisons.shape, objects.shape)
        comparisons = torch.reshape(comparisons, (-1, plan_num, comparisons_shp[-1]))  ##100 * 10 * D
        objects_shp = objects.shape
        objects = torch.reshape(objects, (-1, plan_num, objects_shp[-1]))

        print('predict new comparison, objects: ', comparisons.shape, objects.shape)

        print('predict new comparison sum ', torch.sum(comparisons, 1).shape, torch.sum(comparisons, 1)[:3])

        ###reshape into batch_size x number_of_plan_per_query x feature_size
        context = torch.sum(comparisons, 1, keepdim=True)

        comparison_sums = context - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[1])
        with_query_enc = torch.cat((objects, comparison_sums), 2)

        output = self.output_net(with_query_enc)
        output = torch.squeeze(output, -1)
        return output

    def online_predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # print('start prepare_trees')
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)

        flat_trees, indexes = trees
        # print('Online NN flat_trees shp: ', flat_trees.shape)
        # print('Online NN indexes: ', indexes.shape)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        # print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])

        context = torch.sum(comparisons, 0, keepdim=True)  #
        # print('predict context shp: ', context.shape) # torch.Size([1, 16])

        comparison_sums = context - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('predict new comparison_sums shp: ', comparison_sums.shape) #torch.Size([240, 16])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        output = self.output_net(with_query_enc)
        # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])

        return output



class LTRankNet0OnlyAtten(torch.nn.Module):
    """
    use attention mechanism as weights

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0OnlyAtten, self).__init__()
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
            torch.nn.Linear(16, 64),
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

        # with_query_enc = torch.cat((objects, comparison_sums), 2)

        with_query_enc = copy.copy(comparison_sums)
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

        # with_query_enc = torch.cat((objects, comparison_sums), 2)
        with_query_enc = copy.copy(comparison_sums)

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

        # with_query_enc = torch.cat((objects, comparison_sums),1)
        with_query_enc = copy.copy(comparison_sums)


        output = self.output_net(with_query_enc)
        # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])

        return output


    # def feature_extract(self, query_enc, sample_trees):
    #     assert len(sample_trees) > 1
    #     query_enc = torch.Tensor(query_enc)#*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
    #     query = self.query_net(query_enc)
    #     print('hahahyeye')
    #     trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
    #
    #     flat_trees, indexes = trees
    #     print('Online NN flat_trees shp: ', flat_trees.shape)
    #     print('Online NN indexes: ', indexes.shape)
    #
    #     objects = self.object_net(trees)#.repeat(1,len(sample_trees))
    #     comparisons = self.comparison_net(trees)
    #
    #     print('predict old comparison, objects shp: ', comparisons.shape, objects.shape) #  torch.Size([240, 16]) torch.Size([240, 32])
    #
    #     comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)
    #
    #     print('online attention results: ', comparison_sums[:5], attention_weights[:5])
    #
    #
    #     with_query_enc = torch.cat((objects, comparison_sums),1)
    #
    #     print('extracted features: ', with_query_enc)
    #
    #     # output = self.output_net(with_query_enc)
    #     # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])
    #
    #     return with_query_enc


class LTRankNet0MultiHead(torch.nn.Module):
    """
    use attention mechanism as weights

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0MultiHead, self).__init__()
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

        n_feature_transformer = 16


        self.attn = MultiHeadedAttention(h=4, d_model=n_feature_transformer, dropout=0.1)

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
        comparison_sums = self.attn(comparisons, comparisons, comparisons)

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

        comparison_sums = self.attn(comparisons, comparisons, comparisons)

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

        comparison_sums = self.attn(comparisons, comparisons, comparisons)

        # print('online attention results: ', comparison_sums[:5], attention_weights[:5])

        with_query_enc = torch.cat((objects, comparison_sums),1)

        output = self.output_net(with_query_enc)
        # print('online predict output shp: ', output.shape)  # torch.Size([240, 1])

        return output




class LeroNet(torch.nn.Module):
    def __init__(self, input_feature_dim) -> None:
        super(LeroNet, self).__init__()
        self.input_feature_dim = input_feature_dim
        self._cuda = False
        self.device = None

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.input_feature_dim, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, query_enc, samples_tree, plan_num=10):
        trees = prepare_trees_plans_only(samples_tree, get_features, get_left_child, get_right_child, cuda=query_enc.is_cuda)
        tree_conv = self.tree_conv(trees)
        print('tree_conv shp: ', tree_conv.shape)
        tree_conv_shp = tree_conv.shape
        objects = torch.reshape(tree_conv, (-1, plan_num, tree_conv_shp[-1]))
        output = torch.squeeze(objects, -1)
        print('output shape: ', output.shape)

        return output

    def predict_all(self, query_enc, samples_tree, plan_num=10):
        trees = prepare_trees_plans_only(samples_tree, get_features, get_left_child, get_right_child, cuda=query_enc.is_cuda)
        tree_conv = self.tree_conv(trees)
        print('valid tree_conv shp: ', tree_conv.shape)
        tree_conv_shp = tree_conv.shape
        objects = torch.reshape(tree_conv, (-1, plan_num, tree_conv_shp[-1]))
        output = torch.squeeze(objects, -1)
        print('valid output shape: ', output.shape)

        return output


    def online_predict_all(self, query_enc, samples_tree):
        trees = prepare_trees_plans_only(samples_tree, get_features, get_left_child, get_right_child, cuda=False)
        tree_conv = self.tree_conv(trees)
        print('online tree_conv shp: ', tree_conv.shape)

        print('online output shape: ', tree_conv.shape)

        return tree_conv


"""
transfer learning skills:

log1p for cardinalities and costs
standardization using training-set mean/std
robust scaling using median/IQR
per-query normalization among candidate plans
"""


