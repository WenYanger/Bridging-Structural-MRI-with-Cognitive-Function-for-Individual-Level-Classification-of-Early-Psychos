import torch
import torch.nn as nn
from torch.nn import functional as F
from Model.components.backbone import Backbone
from Model.components.backbone_extractor import Backbone_Extractor
from Model.model import *


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

                
if __name__ == '__main__':
    model = CIBM_Cognitive_3DCNN_CLSREG()
    x = torch.rand(1, 3, 256, 256)
