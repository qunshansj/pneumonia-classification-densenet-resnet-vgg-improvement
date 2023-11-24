
import torch.nn as nn
import torch
class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(      #分类网络结构
            nn.Dropout(p=0.5),                 #50%失活，减少过拟合
            nn.Linear(512*7*7, 2048),          #第一层全连接层，原论文是4096
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, class_num)
        )
        if init_weights:      #是否初始化
            self._initialize_weights()
    def forward(self, x):     #前向传播
        # N x 3 x 224 x 224
        x = self.features(x)      #先提取特征
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)   #展平处理
        # N x 512*7*7
        x = self.classifier(x)    #分类网络
        return x
    def _initialize_weights(self):    #初始化权重函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)   #xavier初始化方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def make_features(cfg: list):    #提取特征的函数
    layers = []   #定义一个空列表
    in_channels = 3
    for v in cfg:     #遍历配置列表
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v   #改变深度
    return nn.Sequential(*layers)     #非关键字参数传入
cfgs = {    #对应不同配置的网络
    #数字代表卷积核个数，字母代表池化层参数
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def vgg(model_name="vgg16", **kwargs):    #实例化模型
    try:
        cfg = cfgs[model_name]    #传入字典
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)   #1特征，2可变长度的字典变量，包含分类个数和是否初始化
    return model
