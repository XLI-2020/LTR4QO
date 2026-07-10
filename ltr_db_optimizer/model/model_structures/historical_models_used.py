import copy

import torch.nn

from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from ltr_db_optimizer.ext.TreeConvolution.util_feature import prepare_trees

from ltr_db_optimizer.allrank.models.transformer import MultiHeadedAttention, make_transformer

from ltr_db_optimizer.ext.TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm, TreeActivation, DynamicPooling
from ltr_db_optimizer.model.featurizer_dict import get_right_child, get_left_child, get_features
from ltr_db_optimizer.ext.TreeConvolution.util_feature import prepare_trees, prepare_trees_plans_only

from ltr_db_optimizer.allrank.models.transformer import make_transformer, MultiHeadedAttention



# This code snippet was mostly extracted, with some renaming from:
# https://github.com/learnedsystems/BaoForPostgreSQL/blob/master/bao_server/net.py

# transformer_para = {
#       "N": 1,
#       "d_ff": 64,
#       "h": 1,
#       "positional_encoding": None,
#       "dropout": 0.0
#     }



Attention = False

Transformer = True

class AttentionPooling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = torch.nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=0)
        return torch.sum(weights * x, dim=0)

class LTRankModel0(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel0, self).__init__()
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

        # self.object_net = torch.nn.Sequential(
        #     BinaryTreeConv(self.input_dimension_1 + 16, 256),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(256, 128),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(128, 64),
        #     TreeLayerNorm(),
        #     DynamicPooling(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(64),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(32),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(16)
        # )

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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        # self.multihead_attention_net = MultiHeadedAttention()
        h = 8  # number of attention heads
        dropout = 0.1
        n_features = 48

        # self.attn = MultiHeadedAttention(h, n_features, dropout)

        # self.attn  = torch.nn.MultiheadAttention(n_features, h)

        if Transformer:
            self.transformer = make_transformer(N=6, d_ff=2048, h=8, dropout=0.1, n_features=48,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=True)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=False)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel2(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel2, self).__init__()
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

        # self.object_net = torch.nn.Sequential(
        #     BinaryTreeConv(self.input_dimension_1 + 16, 256),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(256, 128),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(128, 64),
        #     TreeLayerNorm(),
        #     DynamicPooling(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(64),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(32),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(16)
        # )

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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        # self.multihead_attention_net = MultiHeadedAttention()


        n_features = 48

        # self.attn = MultiHeadedAttention(h, n_features, dropout)

        # self.attn  = torch.nn.MultiheadAttention(n_features, h)

        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=48,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)


class LTRankModel3(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel3, self).__init__()
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
            # torch.nn.Linear(64, 64),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.LeakyReLU()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        # self.multihead_attention_net = MultiHeadedAttention()


        n_features = 64

        # self.attn = MultiHeadedAttention(h, n_features, dropout)

        # self.attn  = torch.nn.MultiheadAttention(n_features, h)

        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        # comparisons = self.comparison_net(trees)
        #
        # print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        # comparisons = self.comparison_net(trees)
        # print(comparisons)
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel3v1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel3v1, self).__init__()
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
            # torch.nn.Linear(64, 64),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.LeakyReLU()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        # self.multihead_attention_net = MultiHeadedAttention()


        n_features = 64

        # self.attn = MultiHeadedAttention(h, n_features, dropout)

        # self.attn  = torch.nn.MultiheadAttention(n_features, h)

        if Transformer:
            self.transformer = make_transformer(N=4, d_ff=1024, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        # comparisons = self.comparison_net(trees)
        #
        # print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        # comparisons = self.comparison_net(trees)
        # print(comparisons)
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel4(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel4, self).__init__()
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
            # torch.nn.Linear(64, 64),
            # torch.nn.LeakyReLU(),
            # # torch.nn.LayerNorm(64),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel5(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel5, self).__init__()
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



        # self.comparison_net = torch.nn.Sequential(
        #     BinaryTreeConv(self.input_dimension_1 + 16, 256),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(256, 128),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(128, 64),
        #     TreeLayerNorm(),
        #     DynamicPooling(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(64),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(32),
        #     torch.nn.Linear(32, 16),
        #     torch.nn.LeakyReLU(),
        #     # torch.nn.LayerNorm(16)
        # )

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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        # self.multihead_attention_net = MultiHeadedAttention()


        n_features = 64

        # self.attn = MultiHeadedAttention(h, n_features, dropout)

        # self.attn  = torch.nn.MultiheadAttention(n_features, h)

        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel6HM(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel6HM, self).__init__()
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

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query,
                              cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))
        comparisons = self.comparison_net(trees)
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        with_query_enc = torch.cat((objects, comparisons), 1)

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
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        with_query_enc = torch.cat((objects, comparisons), 1)
        return self.output_net(with_query_enc)


class LTRankModel7(torch.nn.Module):
    """
    query net + object net/comparison net + transformer struc. optimize.

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel7, self).__init__()
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


        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=512, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        with_query_enc = copy.copy(objects)


        if Transformer:

            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))


        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)


class LTRankModel7v1(torch.nn.Module):
    """
    query net + object net/comparison net + transformer struc. optimize.

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel7v1, self).__init__()
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


        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=6, d_ff=512, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        with_query_enc = copy.copy(objects)


        if Transformer:

            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))


        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel7v2(torch.nn.Module):
    """
    query net + object net/comparison net + transformer struc. optimize.

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel7v2, self).__init__()
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


        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=512, h=8, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        with_query_enc = copy.copy(objects)


        if Transformer:

            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))


        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel7v3(torch.nn.Module):
    """
    query net + object net/comparison net + transformer struc. optimize.

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel7v3, self).__init__()
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
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.LeakyReLU()
        )


        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 64


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=8, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        with_query_enc = copy.copy(objects)


        if Transformer:

            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))


        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)


class LTRankModel7v4(torch.nn.Module):
    """
    query net + object net/comparison net + transformer struc. optimize.

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel7v4, self).__init__()
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
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.LeakyReLU()
        )


        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 64


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=8, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        with_query_enc = copy.copy(objects)


        if Transformer:

            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))


        with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)


class LTRankModel8(torch.nn.Module):
    """
    single road: comparison_net + transformer

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel8v1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8v1, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=8, d_ff=512, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel8v2(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8v2, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=8, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel8v3(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8v3, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 64


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel8v4(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8v4, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            # torch.nn.Linear(16, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel8v5(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8v5, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(32, 32),
            # torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 64


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.2, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)

class LTRankModel8v6(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel8v6, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

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
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU()
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.2, n_features=n_features,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )

    # def get_input_channels(self):
    #    return self.input_channels

    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', with_query_enc.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            with_query_enc = self.transformer(with_query_enc)

            print('after transformer!', with_query_enc.shape)


        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        # objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)
        print(comparisons)
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        with_query_enc = copy.copy(comparison_sums)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            with_query_enc = self.transformer(with_query_enc)




        return self.output_net(with_query_enc)


class LTRankModel9(torch.nn.Module):

    """
    concat: object_net + comparison_net + concatenate + transformer

    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 48

        n_feature_transformer = 16



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)


class LTRankModel9v1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v1, self).__init__()
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
            # # torch.nn.LayerNorm(64),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 96

        n_feature_transformer = 64



        if Transformer:
            self.transformer = make_transformer(N=4, d_ff=2048, h=8, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v2(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v2, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v3(torch.nn.Module):
    """

    based on LTRankModel9_2
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v3, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=6, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v4(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v4, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        print("this model's name: ", self.__class__.__name__)
        print('wo shi da sha zi!')


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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=512, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v5(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v5, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=8, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v6(torch.nn.Module):
    """
    based on LTRankModel9_2
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v6, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=1, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v7(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v7, self).__init__()
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
            # torch.nn.Linear(64, 64),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(64, 64),
            # torch.nn.LeakyReLU(),
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel9v8(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel9v8, self).__init__()
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
            # torch.nn.LayerNorm(64),
            # torch.nn.Linear(64, 32),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 96

        n_feature_transformer = 64



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel10(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel10, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', objects.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            objects = self.transformer(objects)

            print('after transformer!', objects.shape)

        with_query_enc = torch.cat((objects, comparisons), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            objects = self.transformer(objects)

        with_query_enc = torch.cat((objects, comparisons), 1)





        return self.output_net(with_query_enc)

class LTRankModel10v1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel10v1, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32



        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', objects.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            objects = self.transformer(objects)

            print('after transformer!', objects.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            objects = self.transformer(objects)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)


class LTRankModel11(torch.nn.Module):
    """
    cause timeout on tpch-d datasets...
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel11, self).__init__()
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
        print('comparisons shape: ', comparisons.shape)

        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0)
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

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

        print('comparisons shape: ', comparisons.shape)
        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0)
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankModel11v1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel11v1, self).__init__()
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

        print('comparisons shape: ', comparisons.shape)


        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """
        comparison_sums = torch.sum(comparisons, 0)/ comparisons.shape[0]
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)


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

        print('comparisons shape: ', comparisons.shape)

        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0) / comparisons.shape[0]
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankModel11v2(torch.nn.Module):
    """

    concatenate more than 1 comparison_net + object_net

    use sum or mean as context vector
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel11v2, self).__init__()
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

        self.comparison_net2 = torch.nn.Sequential(
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

        print('comparisons shape: ', comparisons.shape)


        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0)
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

        comparisons2 = self.comparison_net2(trees)
        comparison_sums2 = torch.sum(comparisons2, 0)
        comparison_sums2 = comparison_sums2.repeat(comparisons.shape[0], 1)




        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums2), 1)
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

        print('comparisons shape: ', comparisons.shape)
        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0)
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

        comparisons2 = self.comparison_net2(trees)
        comparison_sums2 = torch.sum(comparisons2, 0)
        comparison_sums2 = comparison_sums2.repeat(comparisons.shape[0], 1)

        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums2), 1)

        return self.output_net(with_query_enc)

class LTRankModel11v3(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel11v3, self).__init__()
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

        self.comparison_net2 = torch.nn.Sequential(
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

        self.comparison_net3 = torch.nn.Sequential(
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
            torch.nn.Linear(16 + 32 + 16 + 16, 64),
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

        print('comparisons shape: ', comparisons.shape)


        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0)
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

        comparisons2 = self.comparison_net2(trees)
        comparison_sums2 = torch.sum(comparisons2, 0)
        comparison_sums2 = comparison_sums2.repeat(comparisons.shape[0], 1)

        comparisons3 = self.comparison_net3(trees)
        comparison_sums3 = torch.sum(comparisons3, 0)
        comparison_sums3 = comparison_sums3.repeat(comparisons.shape[0], 1)




        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums2, comparison_sums3), 1)
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

        print('comparisons shape: ', comparisons.shape)
        """
        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])
        """

        comparison_sums = torch.sum(comparisons, 0)
        print('comparison_sums shape after summing: ', comparison_sums)
        comparison_sums = comparison_sums.repeat(comparisons.shape[0], 1)
        print('comparison_sums after repeating vectors: ', comparison_sums.shape, comparison_sums)

        comparisons2 = self.comparison_net2(trees)
        comparison_sums2 = torch.sum(comparisons2, 0)
        comparison_sums2 = comparison_sums2.repeat(comparisons.shape[0], 1)

        comparisons3 = self.comparison_net3(trees)
        comparison_sums3 = torch.sum(comparisons3, 0)
        comparison_sums3 = comparison_sums3.repeat(comparisons.shape[0], 1)

        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums2, comparison_sums3), 1)

        return self.output_net(with_query_enc)


class LTRankModel12(torch.nn.Module):
    """
    to create wider network with original HM model
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel12, self).__init__()
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

        self.comparison_net2 = torch.nn.Sequential(
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
        print('comparisons shape: ', comparisons.shape)


        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])

        comparisons2 = self.comparison_net2(trees)
        comparison_sums2 = torch.sum(comparisons2, 0) - comparisons2
        comparison_sums2 = comparison_sums2 / (comparison_sums2.shape[0])



        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums2), 1)
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

        print('comparisons shape: ', comparisons.shape)

        comparison_sums = torch.sum(comparisons, 0) - comparisons
        comparison_sums = comparison_sums / (comparison_sums.shape[0])

        comparisons2 = self.comparison_net2(trees)
        comparison_sums2 = torch.sum(comparisons2, 0) - comparisons2
        comparison_sums2 = comparison_sums2 / (comparison_sums2.shape[0])


        with_query_enc = torch.cat((objects, comparison_sums, comparison_sums2), 1)
        return self.output_net(with_query_enc)

class LTRankModel13(torch.nn.Module):
    """
    based on LTRankModel9_2
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel13, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel13v1(torch.nn.Module):
    """
    based on LTRankModel9_2
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel13v1, self).__init__()
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32


        if Transformer:
            self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
                     positional_encoding = None)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            comparison_sums = self.transformer(comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            comparison_sums = self.transformer(comparisons)

        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel14(torch.nn.Module):

    """

    only modify the dividend from "n" to "n-1"
    """
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel14, self).__init__()
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
        comparison_sums = comparison_sums / (comparison_sums.shape[0]-1)
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
        comparison_sums = comparison_sums / (comparison_sums.shape[0]-1)
        with_query_enc = torch.cat((objects, comparison_sums), 1)
        return self.output_net(with_query_enc)

class LTRankModel15(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel15, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        print("current model's name: ", self.__class__.__name__)
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32


        if Transformer:
            # self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
            #          positional_encoding = None)
            self.attn = torch.nn.MultiheadAttention(n_feature_transformer, num_heads=4, dropout=0.1)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )




    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums, _ = self.attn(comparisons, comparisons, comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums, _ = self.attn(comparisons, comparisons, comparisons)


        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel15v1(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel15v1, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        print("current model's name: ", self.__class__.__name__)

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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32


        if Transformer:
            # self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
            #          positional_encoding = None)
            # self.attn = torch.nn.MultiheadAttention(n_feature_transformer, num_heads=4, dropout=0.1)
            self.attn = MultiHeadedAttention(h=4, d_model=n_feature_transformer, dropout=0.1)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )




    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums = self.attn(comparisons, comparisons, comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums = self.attn(comparisons, comparisons, comparisons)


        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel15v2(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel15v2, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


        # self.object_net = torch.nn.Sequential(
        #     BinaryTreeConv(self.input_dimension_1 + 16, 512),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(512, 256),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(256, 128),
        #     TreeLayerNorm(),
        #     TreeActivation(torch.nn.LeakyReLU()),
        #     BinaryTreeConv(128, 64),
        #     TreeLayerNorm(),
        #     DynamicPooling(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(32, 32),
        #     torch.nn.LeakyReLU()
        # )

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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            # torch.nn.Linear(32, 16),
            # torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        n_features = 64

        n_feature_transformer = 32


        if Transformer:
            # self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
            #          positional_encoding = None)
            self.attn = torch.nn.MultiheadAttention(n_feature_transformer, num_heads=4, dropout=0.1)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )


    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)

        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums, _ = self.attn(comparisons, comparisons, comparisons)

            print('after transformer!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)


        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums, _ = self.attn(comparisons, comparisons, comparisons)


        with_query_enc = torch.cat((objects, comparison_sums), 1)





        return self.output_net(with_query_enc)

class LTRankModel15v3(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel15v3, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6

        print("current model's name: ", self.__class__.__name__)

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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )


        n_features = 32 + 16

        n_feature_transformer = 16


        if Transformer:
            # self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
            #          positional_encoding = None)
            # self.attn = torch.nn.MultiheadAttention(n_feature_transformer, num_heads=4, dropout=0.1)
            self.attn = MultiHeadedAttention(h=4, d_model=n_feature_transformer, dropout=0.1)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )




    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)


        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)

        query_atten = query.repeat(comparisons.shape[0], 1)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer q=query!', comparisons.shape)

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums = self.attn(query_atten, comparisons, comparisons)

            print('after transformer q=query!', comparison_sums.shape)

        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees(sample_trees, get_features, get_left_child, get_right_child, query, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)
        query_atten = query.repeat(comparisons.shape[0], 1)



        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums = self.attn(query_atten, comparisons, comparisons)


        with_query_enc = torch.cat((objects, comparison_sums), 1)



        return self.output_net(with_query_enc)

class LTRankModel15v4(torch.nn.Module):
    def __init__(self, in_dim_1, in_dim_2):
        super(LTRankModel15v4, self).__init__()
        self.input_dimension_1 = in_dim_1  # Dimension of the Tree Convolution Layers, e.g., 10
        self.input_dimension_2 = in_dim_2  # Dimension of the Query Encoding: e.g, 6


        self.object_net = torch.nn.Sequential(
            BinaryTreeConv(self.input_dimension_1, 512),
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
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )

        self.query_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dimension_2, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16)
        )



        n_features = 32 + 16

        n_feature_transformer = 16


        if Transformer:
            # self.transformer = make_transformer(N=2, d_ff=2048, h=4, dropout=0.1, n_features=n_feature_transformer,
            #          positional_encoding = None)
            # self.attn = torch.nn.MultiheadAttention(n_feature_transformer, num_heads=4, dropout=0.1)
            self.attn = MultiHeadedAttention(h=4, d_model=n_feature_transformer, dropout=0.1)

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            # torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 1)
        )




    def forward(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        print('forward query_enc.is_cuda', query_enc.is_cuda)



        trees = prepare_trees_plans_only(sample_trees, get_features, get_left_child, get_right_child, cuda=query_enc.is_cuda)
        print('type of trees', type(trees), trees)

        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        print('comparisons', comparisons.shape)  # A x16
        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # print('comparison_sums shp', comparison_sums.shape)  # Ax16
        # with_query_enc = torch.cat((objects, comparison_sums), 1)
        # print('with_query_enc shp', with_query_enc.shape) # AX48

        # with_query_enc = copy.copy(objects)

        query_atten = query.repeat(comparisons.shape[0], 1)


        if Transformer:
            # print('enter attention fit !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)
            print('enter transformer in training!', comparisons.shape, comparisons[:5])

            # with_query_enc, attn_output_weights = self.transformer(with_query_enc)

            # comparison_sums = self.transformer(comparisons)

            comparison_sums = self.attn(query_atten, comparisons, comparisons)

            print('after transformer in training!', comparison_sums.shape, comparison_sums[:5])


        with_query_enc = torch.cat((objects, comparison_sums), 1)

        return self.output_net(with_query_enc)

    def predict_all(self, query_enc, sample_trees):
        assert len(sample_trees) > 1
        query_enc = torch.Tensor(query_enc)  # *len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        query = self.query_net(query_enc)
        # query_enc = torch.Tensor([query_enc]*len(sample_trees))#.unsqueeze(1).repeat(1,len(sample_trees))
        print('predict query_enc.is_cuda', query_enc.is_cuda)
        trees = prepare_trees_plans_only(sample_trees, get_features, get_left_child, get_right_child, cuda=query_enc.is_cuda)
        objects = self.object_net(trees)  # .repeat(1,len(sample_trees))

        comparisons = self.comparison_net(trees)

        # comparison_sums = torch.sum(comparisons, 0) - comparisons
        # comparison_sums = comparison_sums / (comparison_sums.shape[0])
        # with_query_enc = torch.cat((objects, comparison_sums), 1)

        # with_query_enc = copy.copy(objects)
        query_atten = query.repeat(comparisons.shape[0], 1)



        if Transformer:
            # print('enter attention predict !!!')
            # print('with_query_enc shape', with_query_enc.shape)
            #
            # with_query_enc = with_query_enc.unsqueeze(0)
            # print('with_query_enc shape after unsqueese', with_query_enc.shape)
            #
            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)
            #
            # print('with_query_enc after attention', with_query_enc.shape)
            # with_query_enc = with_query_enc.squeeze(0)

            # with_query_enc = self.attn(with_query_enc, with_query_enc, with_query_enc)

            # comparison_sums = self.transformer(comparisons)
            print('enter transformer in prediction!', comparisons.shape, comparisons[:5])

            comparison_sums = self.attn(query_atten, comparisons, comparisons)

            print('after transformer in prediction!', comparison_sums.shape, comparison_sums[:5])



        with_query_enc = torch.cat((objects, comparison_sums), 1)



        return self.output_net(with_query_enc)