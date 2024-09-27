# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        #self.qkv = nn.Linear(288, 288 * 3, bias=True)
        setattr(self, "pose", nn.Conv2d(288*2, 288, 3, stride, 1))
        setattr(self,"qkv" ,nn.Linear(120,120 * 3))
        #self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        #setattr(self, "squeeze",nn.Conv2d(576, 256, 1))
        #self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        setattr(self,"pose_0",nn.Conv2d(288, 256, 3, stride, 1))

        #self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        setattr(self, "pose_1", nn.Conv2d(256, 256, 3, stride, 1))

        #self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        setattr(self, "pose_2", nn.Conv2d(256, 6 , 1))
        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))
    def attention(self,q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1)).float())

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, v)
        return output
    def forward(self, input_features):
        #last_features = [f[-1] for f in input_features]
        for i in range(2):
            input_features[i]=getattr(self,"pose")(input_features[i])
        B,C,H,W=input_features[-1].shape
        qkv_linear =getattr(self,"qkv")
        q1, k1, v1= qkv_linear(input_features[0].reshape(B,C,-1)).reshape(B, C, 3, -1).permute(2, 0, 1, 3).contiguous()
        q2, k2, v2 = qkv_linear(input_features[1].reshape(B,C,-1)).reshape(B, C, 3, -1).permute(2, 0, 1, 3).contiguous()
        #cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        #cat_features = [self.relu(getattr(self,"squeeze")(f)) for f in last_features]
        cat_features=self.attention(q1, k2, v1)

        out = cat_features.reshape(B,C,H,W)
        for i in range(3):
            out = getattr(self,"pose_{}".format(i))(out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseDecoder_yg(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder_yg, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        #self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        setattr(self, "net.0",nn.Conv2d(self.num_ch_enc[-1], 256, 1))
        #self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        setattr(self,"net.1",nn.Conv2d(num_input_features * 256, 256, 3, stride, 1))

        #self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        setattr(self, "net.2", nn.Conv2d(256, 256, 3, stride, 1))

        #self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        setattr(self, "net.3", nn.Conv2d(256, 6 * num_frames_to_predict_for, 1))
        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        #cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = [self.relu(getattr(self,"net.0")(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(1,4):
            out = getattr(self,"net.{}".format(i))(out)
            if (i-1) != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseDecoder_y(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder_y, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

