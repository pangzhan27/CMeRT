# version 1, context sample rate is same as long term. now chage it to short-term sample rate
'''
Divide future into short-horizon(anticipation) and long-horizon(future)
--config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml
--test 1 --config_file configs/THUMOS/cmert_long256_work4_kinetics_1x.yaml MODEL.CHECKPOINT checkpoints/THUMOS/cmert_long256_work4_kinetics_1x/epoch-9.pth MODEL.LSTR.INFERENCE_MODE batch
'''
import sys

sys.path.append('./src')
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch.utils.data as data
import os.path as osp
from bisect import bisect_right
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import math
import copy
import os.path as osp
import argparse
import json

from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.utils.ema import build_ema
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.optimizers import build_optimizer
from rekognition_online_action_detection.optimizers import build_scheduler
from rekognition_online_action_detection.evaluation.postprocessing import postprocessing as default_pp
from rekognition_online_action_detection.criterions import build_criterion
from sklearn.metrics import average_precision_score
from rekognition_online_action_detection.utils.registry import Registry

from rekognition_online_action_detection.datasets import build_data_loader, build_dataset
# for models
from rekognition_online_action_detection.models import transformer as tr
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.evaluation import compute_result_new, compute_result


def do_perframe_det_train(cfg,
                          data_loaders,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          ema,
                          device,
                          checkpointer,
                          logger):

    # Setup model on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    s1_w = cfg.SOLVER.get('S1_W', 0.2)
    s2_w = cfg.SOLVER.get('S2_W', 1.0)
    f_w = cfg.SOLVER.get('F_W', 0.5)

    for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.START_EPOCH + cfg.SOLVER.NUM_EPOCHS):
        # Reset
        losses_dict = {}
        for l_name in ['tot', 'ant_cls', 'det_cls', 'fut_cls']:
            losses_dict[l_name] = {phase: 0.0 for phase in cfg.SOLVER.PHASES}
        pred_scores, ant_pred_scores, fut_pred_scores = [], [], []
        gt_targets, ant_gt_targets, fut_gt_targets  = [], [], []

        start = time.time()
        for phase in cfg.SOLVER.PHASES:
            training = phase == 'train'
            model.train(training)
            if not training:
                ema.apply_shadow()

            with torch.set_grad_enabled(training):
                pbar = tqdm(data_loaders[phase],
                            desc='{}ing epoch {}'.format(phase.capitalize(), epoch))
                for batch_idx, data in enumerate(pbar, start=1):
                    batch_size = data[0].shape[0]
                    det_target, ant_target, fut_target = data[-1]

                    loss_names = list(zip(*cfg.MODEL.CRITERIONS))[0][0]
                    scores, fut_scores = model(*[x.to(device) for x in data[:-1]])
                    tot_target = det_target.to(device)
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        tot_target = torch.cat((det_target, ant_target), dim=1).to(device)

                    for i, detant_score in enumerate(scores):
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score, ant_score = detant_score[:, :cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES, :], \
                                                   detant_score[:, cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES:, :]
                            det_score = det_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                            ant_score = ant_score.reshape(-1, cfg.DATA.NUM_CLASSES)

                        detant_score = detant_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        if i == 0:
                            detant_loss = s1_w * criterion[loss_names](detant_score, tot_target.reshape(-1, cfg.DATA.NUM_CLASSES))
                            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                                det_loss = s1_w * criterion['MCE'](det_score, det_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                                ant_loss = s1_w * criterion['MCE'](ant_score, ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                        else:
                            detant_loss += s2_w * criterion[loss_names](detant_score, tot_target.reshape(-1, cfg.DATA.NUM_CLASSES))
                            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                                det_loss += s2_w * criterion['MCE'](det_score,
                                                              det_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                                ant_loss += s2_w * criterion['MCE'](ant_score,
                                                              ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                    fut_loss = torch.FloatTensor([0]).to(scores[0].device)[0]
                    for i, fut_score in enumerate(fut_scores):
                        fut_score = fut_score.reshape(-1, cfg.DATA.NUM_CLASSES)
                        if i == 0:
                            fut_loss = f_w * criterion['MCE'](fut_score, fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))
                        else:
                            fut_loss += f_w * criterion['MCE'](fut_score, fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).to(device))

                    loss = detant_loss + fut_loss
                    losses_dict['tot'][phase] += loss.item() * batch_size
                    losses_dict['fut_cls'][phase] += fut_loss.item() * batch_size
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        losses_dict['ant_cls'][phase] += ant_loss.item() * batch_size
                        losses_dict['det_cls'][phase] += det_loss.item() * batch_size

                    if training:
                        optimizer.zero_grad()
                        if loss.item() != 0:
                            loss.backward()
                            optimizer.step()
                            ema.update()
                            scheduler.step()
                    else:
                        # Prepare for evaluation
                        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                            det_score = det_score.softmax(dim=1).cpu().tolist()
                            det_target = det_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            pred_scores.extend(det_score)
                            gt_targets.extend(det_target)

                            ant_score = ant_score.softmax(dim=1).cpu().tolist()
                            ant_target = ant_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            ant_pred_scores.extend(ant_score)
                            ant_gt_targets.extend(ant_target)
                        else:
                            det_score = detant_score.softmax(dim=1).cpu().tolist()
                            det_target = tot_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            pred_scores.extend(det_score)
                            gt_targets.extend(det_target)

                        if len(fut_scores) > 0 :
                            fut_score = fut_score.softmax(dim=1).cpu().tolist()
                            fut_target = fut_target.reshape(-1, cfg.DATA.NUM_CLASSES).cpu().tolist()
                            fut_pred_scores.extend(fut_score)
                            fut_gt_targets.extend(fut_target)
        end = time.time()

        # Output log for current epoch
        log = []
        log.append('Epoch {:2}'.format(epoch))
        train_log = '[train loss]'
        for k, v in losses_dict.items():
            train_log += ' {}: {:.3f},'.format(k, v['train'] / len(data_loaders['train'].dataset))
        log.append(train_log)

        if 'test' in cfg.SOLVER.PHASES:
            # Compute result
            det_result = compute_result['perframe'](cfg, gt_targets, pred_scores, )
            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                ant_result = compute_result['perframe'](cfg, ant_gt_targets, ant_pred_scores, )
            if len(fut_pred_scores)> 0:
                fut_result = compute_result['perframe'](cfg, fut_gt_targets, fut_pred_scores, )
            test_log = '[test loss]'
            for k, v in losses_dict.items():
                test_log += ' {}: {:.3f},'.format(k, v['test'] / len(data_loaders['test'].dataset))
            log.append(test_log)
            log.append('[mAP] det: {:.3f}， ant: {:.3f}, fut: {:.3f} '.format(det_result['mean_AP'],
                               ant_result['mean_AP'] if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0 else -1,
                                                    fut_result['mean_AP'] if len(fut_pred_scores) >0 else -1))

        log.append('running time: {:.2f} sec'.format(end - start, ))
        logger.info(' | '.join(log))

        # Save checkpoint for model and optimizer
        if epoch % cfg.SOLVER.SAVE_EVERY == 0 and epoch >= 8:
            checkpointer.save(epoch, model, optimizer)

        if not training:
            ema.restore()

        # Shuffle dataset for next epoch
        data_loaders['train'].dataset.shuffle()


def main(cfg):
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='train')
    logger = setup_logger(cfg, phase='train')

    # Build data loaders
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    # Build model
    model = build_model(cfg, device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    logger.info('Number of parameters: {}. Model Size: {:.2f} MB'.format(sum(p.numel() for p in model.parameters()),
                                                                         param_size / 1024 ** 2))

    # Build criterion
    criterion = build_criterion(cfg, device)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Build ema
    ema = build_ema(model, 0.999)

    # Load pretrained model and optimizer
    checkpointer.load(model, optimizer)

    # Build scheduler
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    do_perframe_det_train(
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        ema,
        device,
        checkpointer,
        logger,
    )


######################################################

def do_perframe_det(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='StreamInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect detection scores and targets
    pred_scores = {}
    gt_targets = {}
    ant_scores = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target, _, _ = data[-4]
            score = model(*[x.to(device) for x in data[:-4]])
            score = score[0][-1].softmax(dim=-1).cpu().numpy()

            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        ant_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                    pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            full_indices = torch.cat((query_indices, torch.arange(
                                query_indices[-1] + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE,
                                query_indices[-1] + t_a + 1 + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE)), dim=0)
                            ant_scores[session][full_indices, :, t_a] = score[bs][:full_indices.shape[0]]

                    pred_scores[session][query_indices] = score[bs][:query_indices.shape[0]]
                    gt_targets[session][query_indices] = target[bs]
                else:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            if query_indices[-1] + t_a + 1 < num_frames:
                                ant_scores[session][query_indices[-1] + t_a + 1, :, t_a] = score[bs][
                                    t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]

                    pred_scores[session][query_indices[-1]] = score[bs][-1 - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                    gt_targets[session][query_indices[-1]] = target[bs][-1]

        if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
            maps_list = []
            for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                result = compute_result['perframe'](
                    cfg,
                    np.concatenate(list(gt_targets.values()), axis=0),
                    np.concatenate(list(ant_scores.values()), axis=0)[:, :, t_a],
                )
                logger.info('Action anticipation ({:.2f}s) perframe m{}: {:.5f}'.format(
                    (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                    cfg.DATA.METRICS, result['mean_AP']
                ))
                maps_list.append(result['mean_AP'])
            logger.info('Action anticipation (mean) perframe m{}: {:.5f}'.format(
                cfg.DATA.METRICS, np.mean(maps_list)
            ))
        result = compute_result['perframe'](
            cfg,
            np.concatenate(list(gt_targets.values()), axis=0),
            np.concatenate(list(pred_scores.values()), axis=0),
        )
        logger.info('Action detection perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, result['mean_AP']))

        # import os.path as osp
        # pkl.dump({'cfg': cfg,
        #           'scores': pred_scores,
        #           'labels': gt_targets,
        #           }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.detection.pkl', 'wb'))



def infer(cfg):
    # Setup configurations
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')

    # Build model
    model = build_model(cfg, device)

    # Load pretrained model
    checkpointer.load(model)

    do_perframe_det(
        cfg,
        model,
        device,
        logger,
    )


if __name__ == '__main__':
    cfg = load_cfg()
    if not cfg.TEST:
        main(cfg)
    else:
        infer(cfg)

