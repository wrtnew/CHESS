import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道数为1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(96000, 7),
        )



    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x =x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def embed(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x_fea =x.reshape(x.size(0),-1)
        out = x.view(x.shape[0], -1)
        # out=x
        # print(out.shape)
        return out

    def embed_tucker(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x_fea =x.reshape(x.size(0),-1)
        # out = x.view(x.shape[0], -1)
        # out=x
        # print(out.shape)
        return x

    # def embed_all(self, x):
    #     x1 = self.layer1(x)
    #     x2 = self.layer2(x1)
    #     x3 = self.layer3(x2)
    #     x4 = self.layer4(x3)
    #     print(x1.shape,x2.shape,x3.shape,x4.shape)
    #
    #     # Flatten each feature map
    #     # f1 = x1.view(x1.size(0), -1)
    #     # f2 = x2.view(x2.size(0), -1)
    #     # f3 = x3.view(x3.size(0), -1)
    #     # f4 = x4.view(x4.size(0), -1)
    #     #
    #     # # Concatenate all flattened features
    #     # out = torch.cat([f1, f2, f3, f4], dim=1)
    #     # print(out.shape)
    #
    #     return x1,x2,x3,x4

    def get_feature(self, x, idx_from, idx_to=-1, return_prob=False, return_logit=False):
        if idx_to == -1:
            idx_to = idx_from
        features = []

        for d in range(self.depth):
            x = self.layers['conv'][d](x)
            if self.net_norm:
                x = self.layers['norm'][d](x)
            x = self.layers['act'][d](x)
            if self.net_pooling:
                x = self.layers['pool'][d](x)
            features.append(x)
            if idx_to < len(features):
                return features[idx_from:idx_to + 1]

        if return_prob:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            prob = torch.softmax(logit, dim=-1)
            return features, prob
        elif return_logit:
            out = x.view(x.size(0), -1)
            logit = self.classifier(out)
            return features, logit
        else:
            return features[idx_from:idx_to + 1]

    def forward(self,x,return_features=False):
        x = self.layer1(x)
        x_fea_1 =x
        x = self.layer2(x)
        x_fea_2 =x
        x = self.layer3(x)
        x_fea_3 =x
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)


        if return_features:
            return out, x_fea_1, x_fea_2, x_fea_3
        else:
            return out
        return out
