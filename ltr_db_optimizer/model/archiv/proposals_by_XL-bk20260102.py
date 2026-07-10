import copy

import torch.nn

from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from ltr_db_optimizer.ext.TreeConvolution.util_feature import prepare_trees, prepare_trees_plans_only
# from ltr_db_optimizer.ext.TreeConvolution.util import prepare_trees

from ltr_db_optimizer.allrank.models.transformer import make_transformer, MultiHeadedAttention, attention



Transformer = True

class AttentionPooling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = torch.nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=0)
        return torch.sum(weights * x, dim=0)

class HM(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(HM, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankModel16(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel16, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

        self.comparison_net1 = torch.nn.Sequential(
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
            torch.nn.Linear(16 + 32 + 16, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])

        comparisons1 = self.comparison_net1(trees)
        comparison_sums1 = torch.sum(comparisons1, 0) - comparisons1
        comparison_sums1 = comparison_sums1 / (comparison_sums1.shape[0])



        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums1), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])

        comparisons1 = self.comparison_net1(trees)
        comparison_sums1 = torch.sum(comparisons1, 0) - comparisons1
        comparison_sums1 = comparison_sums1 / (comparison_sums1.shape[0])


        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums1), 1)
        return self.output_net(with_query_enc)

class LTRankNet0(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet0, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        comparison_sums, attention_weights = attention(comparisons, comparisons, comparisons, dropout=0.1)


        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankNet1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet1, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankNet2(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet2, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)


class LTRankNet3(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet3, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        print('comparison[:5]', comparisons[:5])
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('111 comparison_sums[:5]', comparison_sums[:5])



        query_atten = query.repeat(comparisons.shape[0], 1)
        comparison_atten, attention_weights = attention(query_atten, comparisons, comparisons)
        print('comparison_atten[:5] forward', comparison_atten[:5])

        comparison_sums = comparison_atten - comparisons
        print('222 comparison_sums[:5]', comparison_sums[:5])

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """
        query_atten = query.repeat(comparisons.shape[0], 1)
        comparison_atten, attention_weights = attention(query_atten, comparisons, comparisons)
        print('comparison_atten[:5] predict', comparison_atten[:5])

        comparison_sums = comparison_atten - comparisons

        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankNet4v8(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet4v8, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1 + 16, 512),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(512, 256),
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

        # self.query_net = torch.nn.Sequential(
        #     torch.nn.Linear(self.input_dimension_2, 16),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(16, 16),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(16, 16),
        #     torch.nn.LeakyReLU()
        # )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
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

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)

        print('comparison[:5]', comparisons[:5])
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('111 comparison_sums[:5]', comparison_sums[:5])



        query_atten = query.repeat(comparisons.shape[0], 1)
        comparison_atten, attention_weights = attention(query_atten, comparisons, comparisons)
        print('comparison_atten[:5] forward', comparison_atten[:5])

        # comparison_sums = comparison_atten - comparisons
        # print('222 comparison_sums[:5]', comparison_sums[:5])

        with_query_enc = torch.cat((objects, comparison_atten), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)

        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        query_atten = query.repeat(comparisons.shape[0], 1)
        comparison_atten, attention_weights = attention(query_atten, comparisons, comparisons, dropout=0.0)
        print('comparison_atten[:5] predict', comparison_atten[:5])

        # comparison_sums = comparison_atten - comparisons

        with_query_enc = torch.cat((objects, comparison_atten), 1)
        return self.output_net(with_query_enc)


class LTRankNet5(torch.nn.Module):
    """

    HM's comparison net 1
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankNet5, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding

        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1, 256),
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
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.comparison_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
        )

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2 + 16 + 32, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        # query_enc = torch.Tensor([query_enc] * len(sample_trees))  # .unsqueeze(1).repeat(1,len(sample_trees))
        query_enc = torch.Tensor(query_enc)
        query_enc = query_enc.repeat(len(sample_trees), 1)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        with_query_enc = torch.cat((objects, comparison_sums, query_enc), 1)
        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        # query_enc = torch.Tensor([query_enc] * len(sample_trees))  # .unsqueeze(1).repeat(1,len(sample_trees))
        query_enc = torch.Tensor(query_enc)
        query_enc = query_enc.repeat(len(sample_trees), 1)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums/(comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums, query_enc), 1)
        return self.output_net(with_query_enc)

    # def predict_and_get_best(self, samples):
    #    assert len(sample_trees) > 1
    #    query_enc = torch.Tensor(query_enc)
    #    trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child)
    #
    #    results = []
    #    for i in range(len(trees)):
    #        obj = self.object_net(trees[i])
    #        comparison = torch.empty((16,1)) # 16 is from last layer of comparison
    #        for j in range(len(trees)):
    #            if i == j:
    #                continue
    #            comparison += self.comparison_net(trees[j])
    #        results.append(self.output_net(torch.cat(query_enc, obj, comparison)))
    #    return torch.Tensor(results)
