# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn

from rekognition_online_action_detection.utils.registry import Registry

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'obj_ek55_fasterrcnn': 352,
    'rgb_i3d': 1024,
    'rgb_resnet152_audio': 2176,
    'vitg14_rgb_fps24_len6_pos4_stride6_h224_w224': 1536,
    'vitg14_rgb_fps24_len6_pos4_stride6_h336_w336': 1536,
    'vitg14_rgb_fps24_len6_pos4_stride6_h448_w448': 1536,
    'vitg14_rgb_fps24_len6_pos4_stride6_h448_w448_half': 1536,
    'vitg14_rgb_fps24_len6_pos4_stride6_h224_w224_half_crop': 1536,
}


@FEATURE_HEADS.register('THUMOS_TESTRA')
@FEATURE_HEADS.register('EK100_TESTRA')
@FEATURE_HEADS.register('CrossTask_TESTRA')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        if cfg.INPUT.MODALITY in ['visual', 'motion', 'object',
                                  'visual+motion', 'visual+motion+object']:
            self.with_visual = 'visual' in cfg.INPUT.MODALITY
            self.with_motion = 'motion' in cfg.INPUT.MODALITY
            self.with_object = 'object' in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion and self.with_object:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            object_size = FEATURE_SIZES[cfg.INPUT.OBJECT_FEATURE]
            fusion_size = visual_size + motion_size + object_size
        elif self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
        elif self.with_object:
            fusion_size = FEATURE_SIZES[cfg.INPUT.OBJECT_FEATURE]

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            self.input_linear = nn.Sequential(
                nn.Linear(fusion_size, self.d_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_linear = nn.Identity()
        if cfg.MODEL.FEATURE_HEAD.DROPOUT > 0:
            self.dropout = nn.Dropout(cfg.MODEL.FEATURE_HEAD.DROPOUT)
        else:
            self.dropout = None

    def forward(self, visual_input, motion_input, object_input):
        if self.with_visual and self.with_motion and self.with_object:
            fusion_input = torch.cat((visual_input, motion_input,
                                      object_input), dim=-1)
        elif self.with_visual and self.with_motion:
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
        elif self.with_visual:
            fusion_input = visual_input
        elif self.with_motion:
            fusion_input = motion_input
        elif self.with_object:
            fusion_input = object_input
        if self.dropout is not None:
            fusion_input = self.dropout(fusion_input)
        fusion_input = self.input_linear(fusion_input)
        return fusion_input

@FEATURE_HEADS.register('THUMOS')
@FEATURE_HEADS.register('EK100')
@FEATURE_HEADS.register('CrossTask')
class BaseFeatureHead_mat(nn.Module):

    def __init__(self, cfg):
        from rekognition_online_action_detection.models.feature_head import FEATURE_SIZES
        super(BaseFeatureHead_mat, self).__init__()

        MODALITY = cfg.INPUT.MODALITY
        if cfg.INPUT.MODALITY == 'visual+motion':
            MODALITY = 'twostream'
        if MODALITY in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in MODALITY
            self.with_motion = 'visual' not in MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            if self.with_motion:
                self.motion_linear = nn.Sequential(
                    nn.Linear(motion_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
            if self.with_visual:
                self.visual_linear = nn.Sequential(
                    nn.Linear(visual_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Sequential(
                    nn.Linear(2 * self.d_model, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
        else:
            if self.with_motion:
                self.motion_linear = nn.Identity()
            if self.with_visual:
                self.visual_linear = nn.Identity()
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Identity()

    def forward(self, visual_input, motion_input):
        if self.with_visual and self.with_motion:
            visual_input = self.visual_linear(visual_input)
            motion_input  = self.motion_linear(motion_input)
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
            fusion_input = self.input_linear(fusion_input)
        elif self.with_visual:
            fusion_input = self.visual_linear(visual_input)
        elif self.with_motion:
            fusion_input = self.motion_linear(motion_input)

        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)
