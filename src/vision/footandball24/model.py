"""Official fb1 parameter hierarchy, adapted without training or detect-mode dependencies.

Derived from jac99/FootAndBall, commit 01261092a93de55026745e532a2e12188ad86801.
Copyright (c) 2020 Sport Algorithmics and Gaming. MIT notice accompanies this file.
"""
import torch
from torch import nn
from .fpn import FPN,make_modules,cfg


class FootAndBall(nn.Module):
    def __init__(self):
        super().__init__()
        layers,channels=make_modules(cfg['X'],batch_norm=True)
        self.base_network=FPN(layers,channels,32,return_layers=[1,3])
        def head(channels):
            return nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(32,channels,3,padding=1))
        self.ball_classifier=head(2);self.player_classifier=head(2);self.player_regressor=head(4)

    def forward(self,x):
        ball,player=self.base_network(x)
        return {'ball':self.ball_classifier(ball).softmax(1),'player':self.player_classifier(player).softmax(1),
            'player_bbox':self.player_regressor(player)}

