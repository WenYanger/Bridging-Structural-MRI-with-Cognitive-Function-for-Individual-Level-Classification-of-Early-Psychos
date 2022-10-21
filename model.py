import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.components.backbone import Backbone
from Model.components.backbone_extractor import Backbone_Extractor
from Model.model import *

class CIBM_Cognitive_GRU(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_GRU, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels_[-1] * 2, 128, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(128, self.visual_feature_dim, bias=True)
        )
        self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim * 4, 128, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(128, num_cls * 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim * 4, 256, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(256, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x, empty_flag):
        empty_flag = empty_flag[:, :x.shape[2]]

        # Encoder
        # self.e1_, self.e2_, self.e3_, self.e4_ = self.backbone_(x)
        CNN_results = []
        for i in range(x.shape[2]):
            _, _, _, e4_ = self.backbone_(x[:, :, i, :, :])
            CNN_results.append(self.fc1(
                torch.cat([
                    self.avg_pool(e4_).squeeze(-1).squeeze(-1),
                    self.max_pool(e4_).squeeze(-1).squeeze(-1),
                ], -1)))
        CNN_results = torch.stack(CNN_results, 0)

        # GRU_results, h1 = self.gru(CNN_results.float(), torch.randn(self.rnn_num_layer, CNN_results.shape[1], self.rnn_hidden_dim).cuda())
        GRU_results, ht = self.gru(CNN_results.float(), None)

        # Feature Type 1: Mean & max of all time-step
        GRU_results_ = GRU_results.permute(1, 0, 2) * (1 - empty_flag.unsqueeze(-1))
        GRU_results_mean = GRU_results_.sum(1) / (1 - empty_flag).sum(1).unsqueeze(-1)
        GRU_results_max, _ = GRU_results_.max(1)
        final_feature = torch.cat([GRU_results_mean, GRU_results_max], -1)

        # Feature Type 2: Last time-step
        # final_feature = ht[0, :, :]

        logits_cognitive = self.fc_cognitive(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_DNN(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_DNN, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 24
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels_[-1] * 2, 64, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(64, self.visual_feature_dim, bias=True)
        )
        self.fc_dnn = nn.Sequential(
            nn.Linear(self.visual_feature_dim * 158, self.visual_feature_dim * 4, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 4, self.visual_feature_dim, bias=True)
        )
        # self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x, empty_flag):
        empty_flag = empty_flag[:, :x.shape[2]]

        # Encoder
        # self.e1_, self.e2_, self.e3_, self.e4_ = self.backbone_(x)
        CNN_results = []
        for i in range(x.shape[2]):
            _, _, _, e4_ = self.backbone_(x[:, :, i, :, :])
            CNN_results.append(self.fc1(
                torch.cat([
                    self.avg_pool(e4_).squeeze(-1).squeeze(-1),
                    self.max_pool(e4_).squeeze(-1).squeeze(-1),
                ], -1)))
        # CNN_results = torch.stack(CNN_results, 0)
        CNN_results = torch.cat(CNN_results, 1)

        final_feature = self.fc_dnn(CNN_results)

        logits_cognitive = self.fc_cognitive(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_DNN_MultiAngle(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_DNN_MultiAngle, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.slice_num = 110
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels_[-1] * 4 * 4, self.visual_feature_dim * 4, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim * 4),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 4, self.visual_feature_dim, bias=True)
        )
        self.fc_dnn = nn.Sequential(
            nn.Linear(self.visual_feature_dim * self.slice_num, self.visual_feature_dim * 4, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim * 4),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 4, self.visual_feature_dim, bias=True)
        )
        # self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # Encoder
        # self.e1_, self.e2_, self.e3_, self.e4_ = self.backbone_(x)
        CNN_results = []
        for i in range(x.shape[2]):
            _, _, _, e4_ = self.backbone_(x[:, :, i, :, :])
            # CNN_results.append(self.fc1(
            #     torch.cat([
            #         self.avg_pool(e4_).squeeze(-1).squeeze(-1),
            #         self.max_pool(e4_).squeeze(-1).squeeze(-1),
            #     ], -1)))
            CNN_results.append(self.fc1(e4_.view(e4_.size(0), -1)))

        # CNN_results = torch.stack(CNN_results, 0)
        CNN_results = torch.cat(CNN_results, 1)

        final_feature = self.fc_dnn(CNN_results)

        logits_cognitive = self.fc_cognitive(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_DNN_MultiAngle_Pool(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_DNN_MultiAngle_Pool, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 64
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.slice_num = 110
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels_[-1] * 2, self.visual_feature_dim * 2, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim * 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 2, self.visual_feature_dim, bias=True)
        )
        self.fc_dnn = nn.Sequential(
            nn.Linear(self.visual_feature_dim * 2, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True)
        )
        # self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # Encoder
        # self.e1_, self.e2_, self.e3_, self.e4_ = self.backbone_(x)
        CNN_results = []
        for i in range(x.shape[2]):
            _, _, _, e4_ = self.backbone_(x[:, :, i, :, :])
            CNN_results.append(self.fc1(
                torch.cat([
                    self.avg_pool(e4_).squeeze(-1).squeeze(-1),
                    self.max_pool(e4_).squeeze(-1).squeeze(-1),
                ], -1)))
            # CNN_results.append(self.fc1(e4_.view(e4_.size(0), -1)))

        CNN_results_ = torch.stack(CNN_results, 1)
        CNN_results__ = torch.cat([
                torch.max(CNN_results_, dim=1)[0],
                torch.mean(CNN_results_, dim=1),
        ], -1)
        # CNN_results = torch.cat(CNN_results, 1)

        final_feature = self.fc_dnn(CNN_results__)

        logits_cognitive = self.fc_cognitive(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_DNN_MultiAngle_ATT(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_DNN_MultiAngle_ATT, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.slice_num = 128
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels_[-1] * 4 * 4, self.visual_feature_dim * 4, bias=True),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 4, self.visual_feature_dim, bias=True)
        )
        self.fc_dnn = nn.Sequential(
            nn.Linear(self.visual_feature_dim * self.slice_num, self.visual_feature_dim * 4, bias=True),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 4, self.visual_feature_dim, bias=True)
        )
        # self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # Encoder
        # self.e1_, self.e2_, self.e3_, self.e4_ = self.backbone_(x)
        CNN_results = []
        for i in range(x.shape[2]):
            _, _, _, e4_ = self.backbone_(x[:, :, i, :, :])
            # CNN_results.append(self.fc1(
            #     torch.cat([
            #         self.avg_pool(e4_).squeeze(-1).squeeze(-1),
            #         self.max_pool(e4_).squeeze(-1).squeeze(-1),
            #     ], -1)))
            CNN_results.append(self.fc1(e4_.view(e4_.size(0), -1)))
        # CNN_results = torch.stack(CNN_results, 0)
        CNN_results = torch.cat(CNN_results, 1)

        final_feature = self.fc_dnn(CNN_results)

        logits_cognitive = self.fc_cognitive(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_2DCNN_CLSREG(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_2DCNN_CLSREG, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 64
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.slice_num = 110
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels_[-1] * 2, self.visual_feature_dim * 2, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim * 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim * 2, self.visual_feature_dim, bias=True)
        )
        self.fc_dnn = nn.Sequential(
            nn.Linear(self.visual_feature_dim * 2, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True)
        )
        # self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive_CLS = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_cognitive_REG = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        x = torch.cat([x, x[:,0,:,:,:].unsqueeze(1)], 1)
        # Encoder
        # self.e1_, self.e2_, self.e3_, self.e4_ = self.backbone_(x)
        CNN_results = []
        for i in range(x.shape[2]):
            _, _, _, e4_ = self.backbone_(x[:, :, i, :, :])
            CNN_results.append(self.fc1(
                torch.cat([
                    self.avg_pool(e4_).squeeze(-1).squeeze(-1),
                    self.max_pool(e4_).squeeze(-1).squeeze(-1),
                ], -1)))
            # CNN_results.append(self.fc1(e4_.view(e4_.size(0), -1)))

        CNN_results_ = torch.stack(CNN_results, 1)
        CNN_results__ = torch.cat([
                torch.max(CNN_results_, dim=1)[0],
                torch.mean(CNN_results_, dim=1),
        ], -1)
        # CNN_results = torch.cat(CNN_results, 1)

        final_feature = self.fc_dnn(CNN_results__)

        logits_cognitive_CLS = self.fc_cognitive_CLS(final_feature)
        logits_cognitive_REG = self.fc_cognitive_REG(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive_CLS, logits_cognitive_REG, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_DNN_CLSREG(nn.Module):
    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_DNN_CLSREG, self).__init__()

        # if backbone is not None and weight is None:
        #     self.backbone_name = backbone
        #     self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
        #     self.channels_ = self.backbone_.channel_list
        # elif weight is not None and backbone is None:
        #     self.backbone_ = Backbone_Extractor(weight=weight)
        #     self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # Classifier
        self.visual_feature_dim = 1024
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128
        self.slice_num = 110
        # self.fc1 = nn.Sequential(
        #     nn.Linear(self.channels_[-1] * 2, self.visual_feature_dim * 2, bias=True),
        #     nn.BatchNorm1d(self.visual_feature_dim * 2),
        #     nn.Dropout(0.5),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.visual_feature_dim * 2, self.visual_feature_dim, bias=True)
        # )
        self.fc_dnn = nn.Sequential(
            nn.Linear(100, self.visual_feature_dim, bias=True),
            # nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True)
        )
        # self.gru = nn.GRU(input_size=self.visual_feature_dim, hidden_size=self.rnn_hidden_dim, bidirectional=True, num_layers=self.rnn_num_layer)

        self.fc_cognitive_CLS = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            # nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_cognitive_REG = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            # nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            nn.Linear(self.visual_feature_dim, self.visual_feature_dim, bias=True),
            # nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        final_feature = self.fc_dnn(x)

        logits_cognitive_CLS = self.fc_cognitive_CLS(final_feature)
        logits_cognitive_REG = self.fc_cognitive_REG(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive_CLS, logits_cognitive_REG, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_3DCNN(nn.Module):

    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_3DCNN, self).__init__()

        if backbone is not None and weight is None:
            self.backbone_name = backbone
            self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
            self.channels_ = self.backbone_.channel_list
        elif weight is not None and backbone is None:
            self.backbone_ = Backbone_Extractor(weight=weight)
            self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))


        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128

        # 3DCNN
        self.conv_layer1 = self._conv_layver_set(3, 16)
        self.conv_layer2 = self._conv_layver_set(16, 32)
        self.conv_layer3 = self._conv_layver_set(32, 64)
        self.conv_layer4 = self._conv_layver_set(64, 128)
        self.conv_layer5 = self._conv_layver_set(128, 256)
        self.fc_cognitive = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 7 * 2 * 2, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 7 * 2 * 2, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )

        self._init_weight()

    def _conv_layver_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(out_c),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        e1 = self.conv_layer1(x)
        e2 = self.conv_layer2(e1)
        e3 = self.conv_layer3(e2)
        e4 = self.conv_layer4(e3)
        e5 = self.conv_layer5(e4)
        final_feature = e5.view(e5.size(0), -1)
        logits_cognitive = self.fc_cognitive(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_3DCNN_CLSREG(nn.Module):

    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_3DCNN_CLSREG, self).__init__()

        # if backbone is not None and weight is None:
        #     self.backbone_name = backbone
        #     self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
        #     self.channels_ = self.backbone_.channel_list
        # elif weight is not None and backbone is None:
        #     self.backbone_ = Backbone_Extractor(weight=weight)
        #     self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))


        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128

        self.magn_chaneel = 4

        # 3DCNN
        self.conv_layer1 = self._conv_layver_set(2, self.magn_chaneel * 4)
        self.conv_layer2 = self._conv_layver_set(self.magn_chaneel * 4, self.magn_chaneel * 8)
        self.conv_layer3 = self._conv_layver_set(self.magn_chaneel * 8, self.magn_chaneel * 16)
        self.conv_layer4 = self._conv_layver_set(self.magn_chaneel * 16, self.magn_chaneel * 32)
        self.conv_layer5 = self._conv_layver_set(self.magn_chaneel * 32, self.magn_chaneel * 64)
        self.fc_cognitive_CLS = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_cognitive_REG = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 1, bias=True)
        )

        self._init_weight()

    def _conv_layver_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(out_c),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        e1 = self.conv_layer1(x)
        e2 = self.conv_layer2(e1)
        e3 = self.conv_layer3(e2)
        e4 = self.conv_layer4(e3)
        e5 = self.conv_layer5(e4)
        final_feature = e5.view(e5.size(0), -1)
        logits_cognitive_CLS = self.fc_cognitive_CLS(final_feature)
        logits_cognitive_REG = self.fc_cognitive_REG(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive_CLS, logits_cognitive_REG, logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_3DCNN_SZCLS(nn.Module):

    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_3DCNN_SZCLS, self).__init__()

        # if backbone is not None and weight is None:
        #     self.backbone_name = backbone
        #     self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
        #     self.channels_ = self.backbone_.channel_list
        # elif weight is not None and backbone is None:
        #     self.backbone_ = Backbone_Extractor(weight=weight)
        #     self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))


        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128

        # 3DCNN
        self.conv_layer1 = self._conv_layver_set(2, 16)
        self.conv_layer2 = self._conv_layver_set(16, 32)
        self.conv_layer3 = self._conv_layver_set(32, 64)
        self.conv_layer4 = self._conv_layver_set(64, 128)
        self.conv_layer5 = self._conv_layver_set(128, 256)
        self.fc_cognitive_CLS = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, num_cls * 6, bias=True)
        )
        self.fc_cognitive_REG = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 2, bias=True)
        )

        self._init_weight()

    def _conv_layver_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(out_c),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        e1 = self.conv_layer1(x)
        e2 = self.conv_layer2(e1)
        e3 = self.conv_layer3(e2)
        e4 = self.conv_layer4(e3)
        e5 = self.conv_layer5(e4)
        final_feature = e5.view(e5.size(0), -1)
        # logits_cognitive_CLS = self.fc_cognitive_CLS(final_feature)
        # logits_cognitive_REG = self.fc_cognitive_REG(final_feature)
        logits_SZ = self.fc_SZ(final_feature)

        return logits_SZ

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CIBM_Cognitive_3DCNN_CogCLS(nn.Module):

    def __init__(self, backbone=None, weight=None, pretrained=None, num_cls=10):
        super(CIBM_Cognitive_3DCNN_CogCLS, self).__init__()

        # if backbone is not None and weight is None:
        #     self.backbone_name = backbone
        #     self.backbone_ = Backbone(backbone=backbone, pretrained=pretrained)
        #     self.channels_ = self.backbone_.channel_list
        # elif weight is not None and backbone is None:
        #     self.backbone_ = Backbone_Extractor(weight=weight)
        #     self.channels_ = self.backbone_.channels_

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))


        # Classifier
        self.visual_feature_dim = 128
        self.rnn_num_layer = 2
        self.rnn_hidden_dim = 128

        # 3DCNN
        self.conv_layer1 = self._conv_layver_set(2, 16)
        self.conv_layer2 = self._conv_layver_set(16, 32)
        self.conv_layer3 = self._conv_layver_set(32, 64)
        self.conv_layer4 = self._conv_layver_set(64, 128)
        self.conv_layer5 = self._conv_layver_set(128, 256)
        self.fc_cognitive_CLS = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 2, bias=True)
        )
        self.fc_cognitive_REG = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 6, bias=True)
        )
        self.fc_SZ = nn.Sequential(
            # nn.Linear(73 * 30 * 30 * 64, self.visual_feature_dim),
            nn.Linear(256 * 1 * 1 * 1, self.visual_feature_dim),
            nn.BatchNorm1d(self.visual_feature_dim),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(self.visual_feature_dim, 2, bias=True)
        )

        self._init_weight()

    def _conv_layver_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(out_c),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        e1 = self.conv_layer1(x)
        e2 = self.conv_layer2(e1)
        e3 = self.conv_layer3(e2)
        e4 = self.conv_layer4(e3)
        e5 = self.conv_layer5(e4)
        final_feature = e5.view(e5.size(0), -1)
        logits_cognitive_CLS = self.fc_cognitive_CLS(final_feature)
        # logits_cognitive_REG = self.fc_cognitive_REG(final_feature)
        # logits_SZ = self.fc_SZ(final_feature)

        return logits_cognitive_CLS

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = CIBM_Cognitive_3DCNN_CLSREG()
    x = torch.rand(1, 3, 256, 256)
