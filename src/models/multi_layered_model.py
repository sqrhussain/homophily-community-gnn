import torch
import torch.nn.functional as F
import numpy as np


class MonoModel(torch.nn.Module):
    def __init__(self, convType, data, channels, dropout=0.8):
        super(MonoModel, self).__init__()
        self.dropout = dropout
        channels = [data.x.shape[1]] + channels + [len(set([x.item() for x in data.y]))]
        self.conv = []
        for i in range(1, len(channels)):
            conv = convType(channels[i - 1], channels[i])
            self.add_module(str(i), conv)
            self.conv.append(conv)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.conv[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)

        self.embedding = x
        # Last layer
        x = self.conv[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x



class BiModel(torch.nn.Module):
    def __init__(self, convType, dataset, channels, dropout=0.8):
        super(BiModel, self).__init__()
        self.dropout = dropout
        self.conv_st = []
        self.conv_ts = []
        channels_output = [data.x.shape[1]] + [c * 2 for c in channels]
        channels = [data.x.shape[1]] + channels
        for i in range(len(channels) - 1):
            conv_st = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_st' + str(i), conv_st)
            self.conv_st.append(conv_st)

            conv_ts = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_ts' + str(i), conv_ts)
            self.conv_ts.append(conv_ts)

        self.last = convType(channels_output[-1], len(set([x.item() for x in data.y])))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1 - data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        #         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = F.relu(self.conv_st[i](x, st_edges))
            x2 = F.relu(self.conv_ts[i](x, ts_edges))
            x = torch.cat((x1, x2), dim=1)
            x = F.dropout(x, training=self.training, p=self.dropout)

        # last layer
        x = self.last(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


class ASYM(BiModel):

    def __init__(self, convType, dataset, channels, dropout=0.8):
        super(ASYM, self).__init__(convType, dataset, channels, dropout)
        self.last0 = convType(channels[-1]*2, len(set([x.item() for x in data.y])))
        self.last1 = convType(channels[-1]*2, len(set([x.item() for x in data.y])))


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1 - data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        #         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = self.conv_st[i](x, st_edges)
            x2 = self.conv_ts[i](x, ts_edges)
            x = F.relu(torch.cat((x1, x2), dim=1))
            x = F.dropout(x, training=self.training, p=self.dropout)

        # last layer
        x1 = self.last0(x, st_edges)
        x2 = self.last1(x, ts_edges)
        x = x1+x2
        x = F.log_softmax(x, dim=1)

        return x


class TriModel(torch.nn.Module):
    def __init__(self, convType, dataset, channels, dropout=0.8):
        super(TriModel, self).__init__()
        self.dropout = dropout
        self.conv_st = []
        self.conv_ts = []
        self.conv = []
        channels_output = [data.x.shape[1]] + [c * 3 for c in channels]
        channels = [data.x.shape[1]] + channels
        for i in range(len(channels) - 1):
            conv_st = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_st' + str(i), conv_st)
            self.conv_st.append(conv_st)

            conv_ts = convType(channels_output[i], channels[i + 1])
            self.add_module('conv_ts' + str(i), conv_ts)
            self.conv_ts.append(conv_ts)

            conv = convType(channels_output[i], channels[i + 1])
            self.add_module('conv' + str(i), conv)
            self.conv.append(conv)

        self.last = convType(channels_output[-1], len(set([x.item() for x in data.y])))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1 - data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        #         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = F.relu(self.conv_st[i](x, st_edges))
            x2 = F.relu(self.conv_ts[i](x, ts_edges))
            x3 = F.relu(self.conv[i](x, edge_index))
            x = torch.cat((x1, x2, x3), dim=1)
            x = F.dropout(x, training=self.training, p=self.dropout)

        # last layer
        x = self.last(x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x


class pASYM(TriModel):
    def __init__(self, convType, dataset, channels, dropout=0.8):
        super(pASYM, self).__init__(convType, dataset, channels, dropout)
        self.last0 = convType(channels[-1]*3, len(set([x.item() for x in data.y])))
        self.last1 = convType(channels[-1]*3, len(set([x.item() for x in data.y])))
        self.last2 = convType(channels[-1]*3, len(set([x.item() for x in data.y])))


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        st_edges = data.edge_index.t()[1 - data.is_reversed].t()
        ts_edges = data.edge_index.t()[data.is_reversed].t()
        #         print(ts_edges.shape)
        for i in range(len(self.conv_st)):
            x1 = self.conv_st[i](x, st_edges)
            x2 = self.conv_ts[i](x, ts_edges)
            x3 = self.conv[i](x, edge_index)
            x = F.relu(torch.cat((x1, x2, x3), dim=1))
            x = F.dropout(x, training=self.training, p=self.dropout)

        # last layer
        x1 = self.last0(x, st_edges)
        x2 = self.last1(x, ts_edges)
        x3 = self.last2(x, edge_index)
        x = x1+x2+x3
        x = F.log_softmax(x, dim=1)

        return x
