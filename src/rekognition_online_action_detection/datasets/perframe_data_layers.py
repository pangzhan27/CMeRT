# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import copy

from .datasets import DATA_LAYERS as registry
#from rekognition_online_action_detection.utils.ek_utils import (action_to_noun_map, action_to_verb_map)


@registry.register('THUMOS')
@registry.register('CrossTask')
class CMeRTBatchDataLayer(data.Dataset):
    # work feature, (det,ant_target, fut_target) where fut_target is from work_start
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.data_name = cfg.DATA.DATA_NAME
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_num_samples = self.anticipation_length // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        # Future choice
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        # context length
        self.context_length = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS)
        self.context_sample_rate = cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE
        self.context_num_samples = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self.training = phase == 'train'

        self._init_dataset()

    def shuffle(self):
        self._init_dataset()

    def _init_dataset(self):
        self.inputs = []
        if self.data_name.startswith('THUMOS') or self.data_name.startswith('CrossTask'):
            for session in self.sessions:
                target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
                seed = np.random.randint(self.work_memory_length) if self.training else 0
                for work_start, ant_end in zip(range(seed, target.shape[0], self.work_memory_length),
                                               range(seed + self.work_memory_length + self.anticipation_length,
                                                     target.shape[0] - self.anticipation_length, self.work_memory_length)):
                    self.inputs.append([session, work_start, ant_end, target, None, ])

    def segment_sampler(self, start, end, num_samples):
        indices = np.linspace(start, end, num_samples)
        return np.sort(indices).astype(np.int32)

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        (session, work_start, ant_end, target, segments_before) = self.inputs[index]
        visual_inputs = np.load(osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')

        # Get work memory
        work_end = ant_end - self.anticipation_length
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]

        #context_indices = np.arange(work_start-self.context_num_samples,  work_start).clip(0)
        context_indices = np.arange(work_start - self.context_length, work_start).clip(0)
        context_indices = context_indices[::self.context_sample_rate]
        work_indices = np.concatenate((context_indices, work_indices))

        work_visual_inputs = visual_inputs[work_indices]
        work_motion_inputs = motion_inputs[work_indices]

        # Get target
        total_target = copy.deepcopy(target)
        target = target[work_start: ant_end][::self.work_memory_sample_rate]

        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            if self.training:
                long_indices = self.segment_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples).clip(0)
            else:
                long_indices = self.uniform_sampler(
                    long_start,
                    long_end,
                    self.long_memory_num_samples,
                    self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get future
        if self.future_num_samples > 0:
            future_start, future_end = min(work_start, total_target.shape[0] - 1), min(work_start + self.future_num_samples - 1, total_target.shape[0] - 1)
            future_indices = self.future_sampler(future_start, future_end, self.future_num_samples, self.future_sample_rate)
            future_target = total_target[future_indices]
        else:
            future_target = None

        # Get all memory
        if (long_visual_inputs is not None and long_motion_inputs is not None):
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))
        if future_target is not None:
            future_target = torch.as_tensor(future_target.astype(np.float32))
        else:
            future_target = target
        if self.anticipation_num_samples > 0:
            target = (target[:self.work_memory_num_samples], target[self.work_memory_num_samples:], future_target)
        else:
            target = (target[:self.work_memory_num_samples], target, future_target)

        assert fusion_visual_inputs.ndim == fusion_motion_inputs.ndim

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target)

    def __len__(self):
        return len(self.inputs)



@registry.register('StreamInferenceTHUMOS')
@registry.register('StreamInferenceCrossTask')
class CMeRTStreamInferenceDataLayer(data.Dataset):
    # work feature, det_target, work_index

    def __init__(self, cfg, phase='test'):
        self.cfg = cfg
        self.data_root = cfg.DATA.DATA_ROOT
        self.visual_feature = cfg.INPUT.VISUAL_FEATURE
        self.motion_feature = cfg.INPUT.MOTION_FEATURE
        self.object_feature = cfg.INPUT.OBJECT_FEATURE
        self.target_perframe = cfg.INPUT.TARGET_PERFRAME
        self.sessions = getattr(cfg.DATA, phase.upper() + '_SESSION_SET')
        self.long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
        self.long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        self.work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.anticipation_length = cfg.MODEL.LSTR.ANTICIPATION_LENGTH
        self.anticipation_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        self.anticipation_num_samples = self.anticipation_length // self.anticipation_sample_rate
        self.future_length = cfg.MODEL.LSTR.FUTURE_LENGTH
        self.future_sample_rate = cfg.MODEL.LSTR.FUTURE_SAMPLE_RATE
        self.future_num_samples = cfg.MODEL.LSTR.FUTURE_NUM_SAMPLES
        self.context_length = int(cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS)
        self.context_sample_rate = cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE
        self.context_num_samples = int(
            cfg.MODEL.LSTR.CONTEXT_SECONDS * cfg.DATA.FPS) // cfg.MODEL.LSTR.CONTEXT_MEMORY_SAMPLE_RATE

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, self.target_perframe, session + '.npy'))
            for work_start, work_end in zip(
                    range(0, target.shape[0] + 1),
                    range(self.work_memory_length, target.shape[0] + 1)):
                self.inputs.append([
                    session, work_start, work_end,
                    target,
                    target.shape[0]
                ])

    def uniform_sampler(self, start, end, num_samples, sample_rate):
        if start < 0:
            start = (end + 1) % sample_rate
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((np.zeros(padding), indices))
        return np.sort(indices).astype(np.int32)

    def future_sampler(self, start, end, num_samples, sample_rate):
        indices = np.arange(start, end + 1)[::sample_rate]
        padding = num_samples - indices.shape[0]
        if padding > 0:
            indices = np.concatenate((indices, np.full(padding, indices[-1])))[:num_samples]
        assert num_samples == indices.shape[0], f"{indices.shape}"
        return np.sort(indices).astype(np.int32)

    def __getitem__(self, index):
        session, work_start, work_end, target, num_frames = self.inputs[index]
        visual_inputs = np.load(
            osp.join(self.data_root, self.visual_feature, session + '.npy'), mmap_mode='r')
        motion_inputs = np.load(
            osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')


        # Get work memory
        work_indices = np.arange(work_start, work_end).clip(0)
        work_indices = work_indices[::self.work_memory_sample_rate]
        # Get target
        total_target = copy.deepcopy(target)
        target = target[work_indices]

        context_indices = np.arange(work_start - self.context_length, work_start).clip(0)
        context_indices = context_indices[::self.context_sample_rate]
        all_indices = np.concatenate((context_indices, work_indices))
        work_visual_inputs = visual_inputs[all_indices]
        work_motion_inputs = motion_inputs[all_indices]



        # Get long memory
        if self.long_memory_num_samples > 0:
            long_start, long_end = work_start - self.long_memory_length, work_start - 1
            long_indices = self.uniform_sampler(
                long_start,
                long_end,
                self.long_memory_num_samples,
                self.long_memory_sample_rate).clip(0)
            long_visual_inputs = visual_inputs[long_indices]
            long_motion_inputs = motion_inputs[long_indices]

            # Get memory key padding mask
            memory_key_padding_mask = np.zeros(long_indices.shape[0])
            last_zero = bisect_right(long_indices, 0) - 1
            if last_zero > 0:
                memory_key_padding_mask[:last_zero] = float('-inf')
        else:
            long_visual_inputs = None
            long_motion_inputs = None
            memory_key_padding_mask = None

        # Get all memory
        if long_visual_inputs is not None and long_motion_inputs is not None:
            fusion_visual_inputs = np.concatenate((long_visual_inputs, work_visual_inputs))
            fusion_motion_inputs = np.concatenate((long_motion_inputs, work_motion_inputs))
        else:
            fusion_visual_inputs = work_visual_inputs
            fusion_motion_inputs = work_motion_inputs

        future_target = None

        # Convert to tensor
        fusion_visual_inputs = torch.as_tensor(fusion_visual_inputs.astype(np.float32))
        fusion_motion_inputs = torch.as_tensor(fusion_motion_inputs.astype(np.float32))
        if future_target is not None:
            future_target = torch.as_tensor(future_target.astype(np.float32))
        else:
            future_target = target
        target = torch.as_tensor(target.astype(np.float32))
        target = (target, target, future_target)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32))
            return (fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target,
                    session, work_indices, num_frames)
        else:
            return (fusion_visual_inputs, fusion_motion_inputs, fusion_motion_inputs, target,
                    session, work_indices, num_frames)

    def __len__(self):
        return len(self.inputs)



