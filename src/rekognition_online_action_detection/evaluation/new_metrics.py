
from multiprocessing import Pool
from collections import OrderedDict

import numpy as np
import os
import pandas as pd
from sklearn.metrics import average_precision_score
from .postprocessing import postprocessing as default_pp
from rekognition_online_action_detection.utils.registry import Registry

compute_result_new = Registry()

# 1. perframe mAP/ Calibrated mAP
def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def perframe_average_precision(ground_truth,
                               prediction,
                               class_names,
                               ignore_index,
                               metrics,
                               postprocessing):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0, ignore_index])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result['per_class_AP'][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx])
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result

# 2, Perframe evaluation of BG and FG
# 2.1 perframe class-wise mAP
def eval_perframe_map(cfg, ground_truth, prediction, bg_index=0, **kwargs): # the only difference with nobg is remove 0 from ignored_index
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    metrics = kwargs.get('metrics', cfg.DATA.METRICS)
    postprocessing = kwargs.get('postprocessing', default_pp(cfg.DATA.DATA_NAME))

    """Compute (frame-level) average precision between ground truth and
        predictions data frames.
        """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    if ignore_index != 0:
        ignore_index = set([ignore_index])
    else:
        ignore_index = []

        # Compute average precision
    result['per_class_AP'] = OrderedDict()
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result['per_class_AP'][class_name] = compute_score(
                    ground_truth[:, idx], prediction[:, idx])
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))
    result['mean_AP_fg'] = np.mean([v for k, v in result['per_class_AP'].items() if k !=class_names[bg_index]])
    result['mean_AP_bg'] = np.mean([v for k, v in result['per_class_AP'].items() if k == class_names[bg_index]])

    return result

# 2.1 perframe class-wise precision and recall
def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        rankings: numpy ndarray, shape = (instance_count, label_count)
        labels: numpy ndarray, shape = (instance_count,)
        ks: tuple of integers

    Returns:
        list of float: TOP-K accuracy for each k in ks
    """
    if selected_class is not None:
        idx = labels == selected_class
        scores = scores[idx]
        labels = labels[idx]
    rankings = scores.argsort()[:, ::-1]
    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]

def cls_precision(pred, labels, selected_class):
    tp_gt = labels == selected_class
    tp_fp = pred == selected_class
    tp = tp_fp & tp_gt
    return np.sum(tp) / (np.sum(tp_fp) + 1e-8)

def eval_perframe_F1(cfg, ground_truth, prediction, bg_index=0, **kwargs): # we only consider recall with bg
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    postprocessing = kwargs.get('postprocessing', default_pp(cfg.DATA.DATA_NAME))

    result, result1 = OrderedDict(), OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Ignore backgroud class
    if ignore_index != 0 and ignore_index > 0:
        valid_index = np.where(ground_truth[:, ignore_index] != 1)[0]
        ground_truth, prediction =  ground_truth[valid_index], prediction[valid_index]
        ignore_index = [ignore_index]
    else:
        ignore_index = []

    result1['per_class_rec'] = OrderedDict()
    result['per_class_prec'] = OrderedDict()
    labels = np.argmax(ground_truth, axis=-1)
    preds = np.argmax(prediction, axis=-1)
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                result1['per_class_rec'][class_name] = topk_accuracy(prediction, labels, ks=(1,), selected_class=idx)[0]
                result['per_class_prec'][class_name] = cls_precision(preds, labels, selected_class=idx)

    result1['mean_rec'] = np.mean(list(result1['per_class_rec'].values()))
    result1['mean_rec_fg'] = np.mean([v for k, v in result1['per_class_rec'].items() if k != class_names[bg_index]])
    result1['mean_rec_bg'] = np.mean([v for k, v in result1['per_class_rec'].items() if k == class_names[bg_index]])

    result['mean_prec'] = np.mean(list(result['per_class_prec'].values()))
    result['mean_prec_fg'] = np.mean([v for k, v in result['per_class_prec'].items() if k != class_names[bg_index]])
    result['mean_prec_bg'] = np.mean([v for k, v in result['per_class_prec'].items() if k == class_names[bg_index]])

    return result, result1

# 2.2 perframe global accuracy and bg/fg precision and recall
def eval_perframe_acc(cfg, ground_truth, prediction, bg_index=0, **kwargs): # we only consider recall with bg
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    postprocessing = kwargs.get('postprocessing', default_pp(cfg.DATA.DATA_NAME))

    """Compute (frame-level) average precision between ground truth and
        predictions data frames.
        """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Ignore backgroud class
    if ignore_index != 0 and ignore_index > 0:
        valid_index = np.where(ground_truth[:, ignore_index] != 1)[0]
        ground_truth, prediction =  ground_truth[valid_index], prediction[valid_index]

    # accuracy including bg
    labels = np.argmax(ground_truth, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    result['accuracy'] = np.sum(prediction==labels) / len(labels)

    bg_ind = labels == bg_index
    fg_ind = (labels != bg_index)
    result['bg_rec'] = np.sum(prediction[bg_ind] == labels[bg_ind]) / np.sum(bg_ind)
    result['fg_rec'] = np.sum(prediction[fg_ind] == labels[fg_ind]) / np.sum(fg_ind)

    bg_ind1 = prediction == bg_index
    fg_ind1 = (prediction != bg_index)
    result['bg_prec'] = np.sum((prediction[bg_ind1] == labels[bg_ind1]) & (labels[bg_ind1] == bg_index)) / np.sum(bg_ind1)
    result['fg_prec'] = np.sum((prediction[fg_ind1] == labels[fg_ind1]) & (labels[fg_ind1] != bg_index)) / np.sum(fg_ind1)

    return result

# 3. Persegment evaluation of FG
def get_labels_start_end_time(frame_wise_labels, bg_class=[0]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i+1)
    return labels, starts, ends

def f_score(recognized, ground_truth, overlap, bg_class=[0]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    if len(y_label) == 0:
        fp += len(p_label)
        return float(tp), float(fp), 0.0

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def overlap_f1(P, Y, overlap=.1, bg_class=[0]):
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score(P[i], Y[i], overlap, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / float(TP + FP + 1e-8)
    recall = TP / float(TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return precision * 100, recall * 100, F1 * 100

def f_score_ana(recognized, ground_truth, overlap, num_class, bg_class=[0]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = np.zeros(num_class)
    fp = np.zeros(num_class)
    fn = np.zeros(num_class)
    hits = np.zeros(len(y_label))

    if len(y_label) == 0:
        for j in range(len(p_label)):
            fp[p_label[j]] += 1
        return tp, fp, fn

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp[p_label[j]] += 1
            hits[idx] = 1
        else:
            fp[p_label[j]] += 1
    for j in range(len(y_label)):
        if hits[j] == 0:
            fn[y_label[j]] += 1

    return tp, fp, fn

def overlap_f1_macro(P, Y, num_class, overlap=.1, bg_class=["background"]) -> object:
    TP, FP, FN = 0, 0, 0
    for i in range(len(P)):
        tp, fp, fn = f_score_ana(P[i], Y[i], overlap, num_class, bg_class)
        TP += tp
        FP += fp
        FN += fn
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    F1 = np.nan_to_num(F1)
    return precision * 100, recall * 100, F1 * 100

#4. Persegment Point-mAP
def voc_ap(rec, prec, use_07_metric=True, rec_th=1.0):
    """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., rec_th+rec_th/10.0, rec_th/10.0):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    return ap

def point_average_precision(GTs, videoLen, cls, scores, times, videoIds, dist_th, rec_th =1.0, data= 'thumos'):
    """
    inputs:
    GTs: is a dictionary (GTs[videoName][cls] = [AS1, AS2, AS3]). AS represnts the start timestamp of the given cls
    Note GTs[videoIds][0] is for ambiguous class which is ignored
    videoLen: is a dictionary recording video length in seconds. videoLen[videoName] = length in sec
    cls: class of interest (int)
    scores: CAS for all the classes. [\sum_i T_i, #cls(w/ bg)]
    times: per-frame times in seconds[\sum_i T_i]. each video will be counted from 0
    videoIds is the video id of the corresponding confidence and times. [\sum_i T_i]
    dist_th: threshold, (float)
    """
    npos = 0
    R = dict()
    for k, v in enumerate(GTs):
        posct = 0
        for ct in range(len(GTs[v][cls])):
            # print(videoLen[v])
            # print(GTs[v][cls][ct])
            if GTs[v][cls][ct] <= videoLen[v]:
               posct += 1
        npos += posct
        R[v] = [0 for _ in range(len(GTs[v][cls]))]
    confidence = scores[:,cls]
    sorted_ind = np.argsort(-confidence)

    times = times[sorted_ind]
    if data == 'thumos':
        videoIds = ['video_test_'+str(int(videoIds[x])).zfill(7) for x in sorted_ind]
    else:
        videoIds = [videoIds[x] for x in sorted_ind]
    nd = len(videoIds)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        ASs = np.array(GTs[videoIds[d]][cls]).astype(float)
        time = times[d].astype(float)
        dist_min = np.inf
        if len(ASs) > 0:
            # compute absolute distance
            dists = np.abs(time - ASs)
            dist_min = np.min(dists)
            jmin = np.argmin(dists)
        if dist_min <= dist_th:
            if R[videoIds[d]][jmin] == 0:
                tp[d] = 1.
                R[videoIds[d]][jmin] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / (float(npos) +1e-8)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, True, rec_th)
    return rec, prec, ap

def compute_PAP_result(GTs, videoLen, Scores, times, videoIds, dist_th, classnum, ignore=[], rec_th=1.0, data= 'thumos'):
    result = OrderedDict()
    result['pointAP'] = OrderedDict()
    result['mAP'] = OrderedDict()
    for i in range(classnum):
        if not i in ignore:
            rec,prec,result['pointAP'][i] = point_average_precision(GTs, videoLen, i, Scores, times, videoIds, dist_th, rec_th, data)
    result['mAP'] = np.mean(list(result['pointAP'].values()))
    return result

def getASfromCAS(frameScores, videoIds, fps, Ptype, data= 'thumos'):
    '''
       inputs: per-frame scores for all classes (N, class_num);
       corresponding per-frame video Ids;
       fps: frames per second

       outputs: action start scores for all classes (N, class_num);
       corresponding per-frame times in second at its videos;
       length of each video;
    '''
    scores = np.zeros(frameScores.shape)
    times = np.zeros(frameScores.shape[0])
    videoLen = dict()
    # get action start/middle/end from score
    # 1) for each segment, t is the start/mid/end point
    # 2) pred action at t is non-background
    # 3) if 1)&2) hold set action start/mid/end prob = action prob at t
    # 4) otherwise action prob = 0
    preds = np.argmax(frameScores, axis=1)
    l, s, e = get_labels_start_end_time(preds, bg_class=[0])
    for i in range(len(l)):
        ccurr = l[i]
        if Ptype == 'Start':
            pt = s[i]
        elif Ptype == 'Middle':
            pt = (s[i] + e[i] - 1) //2
        elif Ptype == 'End':
            pt = (e[i] - 1)
        scores[pt, ccurr] = frameScores[pt, ccurr]

    previd = videoIds[0]
    counter = 0
    for i in range(0, times.shape[0]):
        currid = videoIds[i]
        if currid != previd:
            counter = 0
            previd = currid
            if data == 'thumos':
                videoLen['video_test_' + str(int(videoIds[i-1])).zfill(7)] = times[i-1]
            else:
                videoLen[videoIds[i-1]] = times[i-1]
        times[i] = counter*1.0/fps
        counter += 1
    # add the last one
    if data == 'thumos':
        videoLen['video_test_'+str(int(videoIds[-1])).zfill(7)] = times[-1]
    else:
        videoLen[videoIds[-1]] = times[-1]
    return scores, times, videoLen

def evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Start', ignore=[0, 21], data= 'thumos'):
    '''
    Point-wise mAP
    Args:
        cfg: configuration dict
        pred_scores: of shape [\sum_i T_i, #class]
        gt_targets: of shape [\sum_i T_i, #class]
        Ptype: Start/Middle/End point detection
        ignore: ignored index list
    Returns:

    '''
    assert Ptype in ['Start', 'Middle', 'End'], 'Wrong P-AP type {}, not in [Start, Middle, End]'.format(Ptype)
    # calculate the GTs
    class_names = cfg.DATA.CLASS_NAMES
    fps = cfg.DATA.FPS
    frameScores, GTs, videoIds = [], {}, []
    for vname in gt_targets.keys():
        vid = vname.split('_')[-1]
        single_score, single_gt = pred_scores[vname], gt_targets[vname]
        if data == 'thumos':
            videoIds.extend([int(vid) for i in range(single_score.shape[0])])
        else:
            videoIds.extend([vname for i in range(single_score.shape[0])])
        frameScores.extend(single_score)
        # prepare GT
        GTs[vname] = []
        assert single_score.shape[1] == len(class_names)
        for cls in range(single_score.shape[1]):
            # ignore GT preparation for ignored cls
            if cls in ignore:
                GTs[vname].append([])
                continue

            cls_points = []
            cls_gt = single_gt[:, cls]

            # no cls in the current video
            if np.max(cls_gt) == 0:
                GTs[vname].append([])
                continue

            l, s, e = get_labels_start_end_time(cls_gt, bg_class=[0])
            for j in range(len(s)):
                if Ptype == 'Start':
                    pt = s[j] / fps
                elif Ptype == 'Middle':
                    pt = (s[j] + e[j]-1)/ (2*fps)
                elif Ptype == 'End':
                    pt = (e[j]-1) / fps
                cls_points.append(pt)
            GTs[vname].append(cls_points)

    scores, times, videoLen = getASfromCAS(np.array(frameScores), videoIds, fps, Ptype, data)
    #scores, times, videoLen = getASfromCAS_start(np.array(frameScores), videoIds, fps)
    dist_ths = [1.0, 2.0, 3.0, 4.0, 5.0] #, 6.0, 7.0, 8.0, 9.0, 10.0
    pAP_1, pAP_mean = 0, 0
    for dist_th in dist_ths:
        result_point = compute_PAP_result(GTs, videoLen, scores, times, videoIds, dist_th, len(class_names), ignore=ignore, data= data)
        #print('Test point mAP @ dist_th = ' + str(dist_th), result_point['mAP'])
        if dist_th == 1:
            pAP_1 = result_point['mAP']
            pAp_1_cls = result_point['pointAP']
        pAP_mean += result_point['mAP']
    return pAP_1, pAP_mean/len(dist_ths), pAp_1_cls



@compute_result_new.register('perframe')
def eval_perframe(cfg, ground_truth, prediction, **kwargs):
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    metrics = kwargs.get('metrics', cfg.DATA.METRICS)
    postprocessing = kwargs.get('postprocessing', default_pp(cfg.DATA.DATA_NAME))
    return perframe_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=class_names,
        ignore_index=ignore_index,
        metrics=metrics,
        postprocessing=postprocessing,
    )

@compute_result_new.register('THUMOS')
def thumos_results(cfg, pred_scores, gt_targets, logger):
    class_names = cfg.DATA.CLASS_NAMES
    # write prediction results
    import os
    output_path = os.path.splitext(cfg.MODEL.CHECKPOINT)[0] + '_{}_pred'.format(cfg.MODEL.LSTR.INFERENCE_MODE)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for session in pred_scores:
        log_file = os.path.join(output_path, '{}.txt'.format(session))
        y = np.argmax(gt_targets[session], axis=-1)
        hat_y = np.argmax(pred_scores[session], axis=-1) # changed to include bg
        valid_ind = (y != cfg.DATA.IGNORE_INDEX) # remove y=0

        y= y[valid_ind]
        hat_y = hat_y[valid_ind]
        stamp = np.arange(len(valid_ind))[valid_ind]

        out_str = ''
        for i in range(len(y)):
            out_str += '%-10s%-20s%-20s \n'% (str(stamp[i]), class_names[y[i]], class_names[hat_y[i]])

        f_ptr = open(log_file, "w")
        f_ptr.write('%-10s%-20s%-20s \n'%('Stamp', 'GT', 'Pred'))
        f_ptr.write(out_str)
        f_ptr.close()

    sess_pred, sess_gt = [], []
    smoothed_pred_score = {}
    from scipy.ndimage import median_filter
    for key in pred_scores.keys():
        # post processing to smooth
        output = pred_scores[key]
        smoothed_output = np.zeros_like(output)
        for c in range(output.shape[1]):
            if c != cfg.DATA.IGNORE_INDEX:
                smoothed_output[:, c] = median_filter(output[:, c], size=cfg.MODEL.PP_SIZE)
        output = smoothed_output / smoothed_output.sum(1, keepdims=True)
        smoothed_pred_score[key] = output
        single_pred = np.argmax(smoothed_output, axis=-1)
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

    # 1. point-mAP
    start_sm_pap1, start_sm_pap_mean, start_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='Start')
    mid_sm_pap1, mid_sm_pap_mean, mid_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='Middle')
    end_sm_pap1, end_sm_pap_mean, end_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='End')

    start_pap1, start_pap_mean, start_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Start')
    mid_pap1, mid_pap_mean, mid_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Middle')
    end_pap1, end_pap_mean, end_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='End')

    # 2. segment F1 score
    f1_ignore_cls = list(set([0, cfg.DATA.IGNORE_INDEX]))

    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # 3. frame-wise results
    all_pred, all_gt = np.concatenate(list(pred_scores.values()), axis=0), np.concatenate(list(gt_targets.values()),
                                                                                          axis=0)
    result_map = eval_perframe_map(cfg, all_gt, all_pred)
    result_prec, result_rec = eval_perframe_F1(cfg, all_gt, all_pred)
    result_acc = eval_perframe_acc(cfg, all_gt, all_pred)

    # show results for each class
    logger.info('############## per class results(mAP, Rec, Prec) ##############')
    for c in class_names:
        if c not in result_map['per_class_AP']:
            ap = -0.01
        else:
            ap = result_map['per_class_AP'][c]

        if c not in result_rec['per_class_rec']:
            rec1 = -0.01
        else:
            rec1 = result_rec['per_class_rec'][c]

        if c not in result_prec['per_class_prec']:
            prec = -0.01
        else:
            prec = result_prec['per_class_prec'][c]

        logger.info('%-20s%-10.1f%-10.1f%-10.1f\n' % (c, ap * 100, rec1 * 100, prec * 100))

    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': 'median' + '_s{}'.format(cfg.MODEL.PP_SIZE),
                  'mAP_fg': result_map['mean_AP_fg'] * 100, 'mAP_bg': result_map['mean_AP_bg'] * 100,
                  'mRec_fg': result_rec['mean_rec_fg'] * 100, 'mRec_bg': result_rec['mean_rec_bg'] * 100,
                  'mPrec_fg': result_prec['mean_prec_fg'] * 100, 'mPrec_bg': result_prec['mean_prec_bg'] * 100,
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_mid@1': mid_pap1 * 100,  'mPAP_mid@1-5': mid_pap_mean * 100,
                  'mPAP_mid@1(p.p)': mid_sm_pap1 * 100, 'mPAP_mid@1-5(p.p)': mid_sm_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10,  'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': result_acc['accuracy'] * 100,
                  '||Extra|| ': ' || ',  # extra results
                  'mPAP_start@1': start_pap1 * 100, 'mPAP_start@1-5': start_pap_mean * 100,
                  'mPAP_start@1(p.p)': start_sm_pap1 * 100, 'mPAP_start@1-5(p.p)': start_sm_pap_mean * 100,
                  'mPAP_end@1': end_pap1 * 100, 'mPAP_end@1-5': end_pap_mean * 100,
                  'mPAP_end@1(p.p)': end_sm_pap1 * 100, 'mPAP_end@1-5(p.p)': end_sm_pap_mean * 100,
                  }

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head, float_format='%.1f')

@compute_result_new.register('CrossTask')
def crosstask_results(cfg, pred_scores, gt_targets, logger):
    class_names = cfg.DATA.CLASS_NAMES
    ## write prediction results
    import os
    output_path = os.path.splitext(cfg.MODEL.CHECKPOINT)[0] + '_{}_pred'.format(cfg.MODEL.LSTR.INFERENCE_MODE)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for session in pred_scores:
        log_file = os.path.join(output_path, '{}.txt'.format(session))
        y = np.argmax(gt_targets[session], axis=-1)
        hat_y = np.argmax(pred_scores[session], axis=-1)  # changed to include bg
        # valid_ind = (y != cfg.DATA.IGNORE_INDEX) # remove y=0
        valid_ind = (y != -100)  # keep background

        y = y[valid_ind]
        hat_y = hat_y[valid_ind]
        stamp = np.arange(len(valid_ind))[valid_ind]

        out_str = ''
        for i in range(len(y)):
            out_str += '%-10s \t %-20s \t %-20s \n' % (str(stamp[i]), class_names[y[i]], class_names[hat_y[i]])

        f_ptr = open(log_file, "w")
        f_ptr.write('%-10s \t %-20s \t %-20s \n' % ('Stamp', 'GT', 'Pred'))
        f_ptr.write(out_str)
        f_ptr.close()

    sess_pred, sess_gt = [], []
    smoothed_pred_score = {}
    from scipy.ndimage import median_filter
    for key in pred_scores.keys():
        # post processing to smooth
        output = pred_scores[key]
        smoothed_output = np.zeros_like(output)
        for c in range(output.shape[1]):
            if c != cfg.DATA.IGNORE_INDEX:
                smoothed_output[:, c] = median_filter(output[:, c], size=cfg.MODEL.PP_SIZE)
        output = smoothed_output / smoothed_output.sum(1, keepdims=True)
        smoothed_pred_score[key] = output
        single_pred = np.argmax(smoothed_output, axis=-1)
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

    # 1. point-mAP
    start_sm_pap1, start_sm_pap_mean, start_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets,
                                                                       Ptype='Start', ignore=[], data='crosstask')
    mid_sm_pap1, mid_sm_pap_mean, mid_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='Middle',
                                                                 ignore=[], data='crosstask')
    end_sm_pap1, end_sm_pap_mean, end_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='End',
                                                                 ignore=[], data='crosstask')

    start_pap1, start_pap_mean, start_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Start', ignore=[], data='crosstask')
    mid_pap1, mid_pap_mean, mid_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Middle', ignore=[], data='crosstask')
    end_pap1, end_pap_mean, end_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='End', ignore=[], data='crosstask')

    # 2. segment F1 score
    if cfg.DATA.IGNORE_INDEX > 0:
        f1_ignore_cls = list(set([0, cfg.DATA.IGNORE_INDEX]))
    else:
        f1_ignore_cls = [0]

    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1,
                                                 bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25,
                                                 bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5,
                                                 bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # 3. frame-wise results
    all_pred, all_gt = np.concatenate(list(pred_scores.values()), axis=0), np.concatenate(list(gt_targets.values()),
                                                                                          axis=0)
    result_map = eval_perframe_map(cfg, all_gt, all_pred)
    result_prec, result_rec = eval_perframe_F1(cfg, all_gt, all_pred)
    result_acc = eval_perframe_acc(cfg, all_gt, all_pred)

    # show results for each class
    logger.info('############## per class results(mAP, Rec, Prec) ##############')
    for c in class_names:
        if c not in result_map['per_class_AP']:
            ap = -0.01
        else:
            ap = result_map['per_class_AP'][c]

        if c not in result_rec['per_class_rec']:
            rec1 = -0.01
        else:
            rec1 = result_rec['per_class_rec'][c]

        if c not in result_prec['per_class_prec']:
            prec = -0.01
        else:
            prec = result_prec['per_class_prec'][c]

        logger.info('%-20s%-10.1f%-10.1f%-10.1f\n' % (c, ap * 100, rec1 * 100, prec * 100))

    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': 'median' + '_s{}'.format(cfg.MODEL.PP_SIZE),
                  'mAP_fg': result_map['mean_AP_fg'] * 100, 'mAP_bg': result_map['mean_AP_bg'] * 100,
                  'mRec_fg': result_rec['mean_rec_fg'] * 100, 'mRec_bg': result_rec['mean_rec_bg'] * 100,
                  'mPrec_fg': result_prec['mean_prec_fg'] * 100, 'mPrec_bg': result_prec['mean_prec_bg'] * 100,
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_mid@1': mid_pap1 * 100, 'mPAP_mid@1-5': mid_pap_mean * 100,
                  'mPAP_mid@1(p.p)': mid_sm_pap1 * 100, 'mPAP_mid@1-5(p.p)': mid_sm_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10, 'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': result_acc['accuracy'] * 100,
                  '||Extra|| ': ' || ',  # extra results
                  'mPAP_start@1': start_pap1 * 100, 'mPAP_start@1-5': start_pap_mean * 100,
                  'mPAP_start@1(p.p)': start_sm_pap1 * 100, 'mPAP_start@1-5(p.p)': start_sm_pap_mean * 100,
                  'mPAP_end@1': end_pap1 * 100, 'mPAP_end@1-5': end_pap_mean * 100,
                  'mPAP_end@1(p.p)': end_sm_pap1 * 100, 'mPAP_end@1-5(p.p)': end_sm_pap_mean * 100,
                  }
    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head, float_format='%.1f')

@compute_result_new.register('THUMOS_mAP')
def thumos_results_map(cfg, pred_scores, gt_targets, logger):
    class_names = cfg.DATA.CLASS_NAMES

    # frame-wise results
    all_pred, all_gt = np.concatenate(list(pred_scores.values()), axis=0), np.concatenate(list(gt_targets.values()),
                                                                                          axis=0)
    result_map = eval_perframe_map(cfg, all_gt, all_pred)
    result_prec, result_rec = eval_perframe_F1(cfg, all_gt, all_pred)

    return result_map['mean_AP_fg'], result_rec['mean_rec_fg'], result_prec['mean_prec_fg']

### point-wise F1
from typing import List, Optional, Tuple, Union
def get_sequence_from_frame_labels(frame_wise_labels, bg_class = None):
    """Collapses a list of frame-wise labels into a list of segments
    Args:
        frame_wise_labels: corresponds to either the GT or predicted sequence of labels
            > e.g., ["pick up", "pick up", "position", "background" , "position", "position", "position", "background", "screw"]
        bg_class: list of background classes, either as label or index, for example ["background"] or [0].

    Returns:
        segments: List of segment labels,
            > e.g., ['pick up', 'position', 'screw']
        segment_starts: stores start frames for each segment
            > e.g.,  [0, 2, 8]
        segment_ends: stores end frames for each segment
            > e.g.,  [1, 6, 8]
    """
    if bg_class is None:
        bg_class = [0]  # Initialize bg_class inside the function if it's not provided

    segment_labels = []
    segment_starts = []
    segment_ends = []

    # set the first segment
    last_segment = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        segment_labels.append(frame_wise_labels[0])
        segment_starts.append(0)

    # loop through all frames to identify segments
    i = 0
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_segment:
            if frame_wise_labels[i] not in bg_class:
                segment_labels.append(frame_wise_labels[i])
                segment_starts.append(i)
            if last_segment not in bg_class:
                segment_ends.append(i)
            last_segment = frame_wise_labels[i]
    if last_segment not in bg_class:
        segment_ends.append(i)

    return segment_labels, segment_starts, segment_ends

def compute_point_level_accuracies(y_true, y_pred, threshold = 0, bg_class = None, dtype = 'Start'):
    """Computes the point level accuracies (f1 score, precision, recall).
    Check if the action has been correctly recognized according to its distance to the action start.
    If the predicted action start is within a certain radius of the ground-truth action start, that
    prediction is considered to be a true positive.

    Reference:
    [1] WOAD: Weakly Supervised Online Action Detection in Untrimmed Videos, Gao et al., CVPR'21
    [2] StartNet: Online Detection of Action Start in Untrimmed Videos, Gao et al., ICCV'19

    Args:
        y_true: corresponds to the ground truth sequence of labels
                > e.g., ["pick up", "pick up", "position", "position"]
        y_pred: corresponds to the predicted sequence of labels
                > e.g., ["pick up", "position", "position", "screw"]
        threshold: the threshold used to determine if the action start is correctly detected
    Returns:
        f1 score: f1 score for the detection accuracy with the given threshold
        precision: precision for the detection accuracy with the given threshold
        recall: recall for the detection accuracy with the given threshold
    """
    eps = 0.0001
    assert dtype in ['Start', 'Mid'], 'unsurported dtype on point-wise F1'

    bg_class = bg_class if bg_class else [0]

    y_label, y_start, y_end = get_sequence_from_frame_labels(y_true, bg_class)
    p_label, p_start, p_end = get_sequence_from_frame_labels(y_pred, bg_class)

    # use mid
    if dtype == 'Mid':
        y_start = [(y_start[i] + y_end[i])/2 for i in range(len(y_start))]
        p_start = [(p_start[i] + p_end[i]) / 2 for i in range(len(p_start))]

    if len(y_label) == 0 and len(p_label) == 0:
        return 0, 0, 0

    # Count true and false positives within the overlapping area
    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        if len(y_start) == 0:
            dists = [np.inf]
        else:
            dists = np.abs(p_start[j] - np.array(y_start))
        dist_min = np.min(dists)
        jmin = np.argmin(dists)

        if dist_min <= threshold:
            if p_label[j] == y_label[jmin]:
                if not hits[jmin]:
                    tp += 1
                    hits[jmin] = 1
                else:
                    fp += 1
            else:
                fp += 1
        else:
            fp += 1

    # Compute the false negative count
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def compute_point_F1(cfg, pred_scores, gt_targets, bg_class=None, dtype = 'Start', dist_ths = [1.0, 2.0, 3.0, 4.0, 5.0]):
    out_results = []
    for dist_th in dist_ths:
        threshold = dist_th * cfg.DATA.FPS
        TP, FP, FN = 0, 0, 0
        for key in gt_targets.keys():
            y_true, y_pred = gt_targets[key], pred_scores[key]
            y_true, y_pred = np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)
            tp, fp, fn = compute_point_level_accuracies(y_true, y_pred, threshold=threshold, bg_class=bg_class, dtype=dtype)
            TP += tp
            FP += fp
            FN += fn

        precision = TP / float(TP + FP + 1e-8)
        recall = TP / float(TP + FN + 1e-8)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        F1 = np.nan_to_num(F1)
        out_results.append([round(F1* 100, 2), round(precision* 100, 2), round(recall* 100, 2)])

    avg_result = np.mean(np.array(out_results), axis=0)

    return np.concatenate((np.array(out_results), avg_result.reshape(1, -1)), axis=0)

### Edit score

def levenstein_(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], 'float')
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / (max(m_row, n_col) + 1e-8)) * 100
    else:
        score = D[-1, -1]

    return score

def edit_score(P, Y, norm=True, bg_class=[]):
    if type(P) == list:
        tmp = [edit_score(P[i], Y[i], norm, bg_class) for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_, _, _ = get_labels_start_end_time(P, bg_class)
        Y_, _, _ = get_labels_start_end_time(Y, bg_class)
        return levenstein_(P_, Y_, norm)


######
def postprocess_results(cfg, pred_scores, gt_targets, pp_size, f1_ignore_cls):
    class_names = cfg.DATA.CLASS_NAMES
    sess_pred, sess_gt = [], []
    smoothed_pred_score = {}
    from scipy.ndimage import median_filter
    for key in pred_scores.keys():
        output = pred_scores[key]
        # post prcessing on prediction. this is for F1-based metric
        single_pred = np.argmax(output, axis=-1)
        single_pred_new = median_filter(single_pred, mode="nearest", size=pp_size)
        pad_start = np.array([single_pred_new[0]]* int(pp_size//2))
        single_pred = np.concatenate((pad_start, single_pred_new[:-int(pp_size//2)]))
        single_gt = np.argmax(gt_targets[key], axis=-1)

        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

        # post processing to smooth on probability, this is done only for map metric
        smoothed_output = np.zeros_like(output)
        for c in range(output.shape[1]):
            if c != cfg.DATA.IGNORE_INDEX:
                smoothed_output_new = median_filter(output[:, c], mode="nearest", size=pp_size)
                pad_output = np.array([smoothed_output_new[0]]* int(pp_size//2))
                smoothed_output[:, c] = np.concatenate((pad_output, smoothed_output_new[:-int(pp_size//2)]))
        smoothed_output = smoothed_output / smoothed_output.sum(1, keepdims=True)
        smoothed_pred_score[key] = smoothed_output

    # 1. point-mAP
    start_sm_pap1, start_sm_pap_mean, start_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, ignore=f1_ignore_cls, Ptype='Start')
    mid_sm_pap1, mid_sm_pap_mean, mid_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, ignore=f1_ignore_cls, Ptype='Middle')
    end_sm_pap1, end_sm_pap_mean, end_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, ignore=f1_ignore_cls, Ptype='End')
    # 1.2 point-F1
    start_pF1_sm = compute_point_F1(cfg, smoothed_pred_score, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1_sm = compute_point_F1(cfg, smoothed_pred_score, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 2. segment F1 score
    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # global
    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)

    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': 'median' + '_s{}'.format(pp_size),
                  'mAP_fg': '-', 'mAP_bg': '-',
                  'mRec_fg': '-', 'mRec_bg': '-',
                  'mPrec_fg': '-', 'mPrec_bg': '-',
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_mid@1': mid_sm_pap1 * 100, 'mPAP_mid@1-5': mid_sm_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10, 'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': '-', 'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50, 'Edit': edit,
                  '||Extra|| ': ' || ',  # extra results
                  'mPAP_start@1': start_sm_pap1 * 100, 'mPAP_start@1-5': start_sm_pap_mean * 100,
                  'mPAP_end@1': end_sm_pap1 * 100, 'mPAP_end@1-5': end_sm_pap_mean * 100,
                  '||Point-F1|| ': ' || ',
                  'Start_1s(F1, prec, rec)': start_pF1_sm[0], 's2s': start_pF1_sm[1], 's3s': start_pF1_sm[2], 's4s': start_pF1_sm[3],
                  's5s': start_pF1_sm[4], 'savg': start_pF1_sm[5],
                  '|| ': ' || ',
                  'Mid_1s(F1, prec, rec)': mid_pF1_sm[0], 'm2s': mid_pF1_sm[1], 'm3s': mid_pF1_sm[2], 'm4s': mid_pF1_sm[3],
                  'm5s': mid_pF1_sm[4], 'mavg': mid_pF1_sm[5],
                  }

    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head,
              float_format='%.1f')


@compute_result_new.register('THUMOS_final')
def thumos_results_final(cfg, pred_scores, gt_targets, logger):
    class_names = cfg.DATA.CLASS_NAMES
    # write prediction results
    import os
    output_path = os.path.splitext(cfg.MODEL.CHECKPOINT)[0] + '_{}_pred'.format(cfg.MODEL.LSTR.INFERENCE_MODE)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for session in pred_scores:
        log_file = os.path.join(output_path, '{}.txt'.format(session))
        y = np.argmax(gt_targets[session], axis=-1)
        hat_y = np.argmax(pred_scores[session], axis=-1) # changed to include bg
        valid_ind = (y != cfg.DATA.IGNORE_INDEX) # remove y=0

        y= y[valid_ind]
        hat_y = hat_y[valid_ind]
        stamp = np.arange(len(valid_ind))[valid_ind]

        out_str = ''
        for i in range(len(y)):
            out_str += '%-10s%-20s%-20s \n'% (str(stamp[i]), class_names[y[i]], class_names[hat_y[i]])

        f_ptr = open(log_file, "w")
        f_ptr.write('%-10s%-20s%-20s \n'%('Stamp', 'GT', 'Pred'))
        f_ptr.write(out_str)
        f_ptr.close()

    f1_ignore_cls = list(set([0, cfg.DATA.IGNORE_INDEX]))
    ## 1. [results w/o post process]
    # 1.1 point-mAP
    start_pap1, start_pap_mean, start_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, ignore=f1_ignore_cls,Ptype='Start')
    mid_pap1, mid_pap_mean, mid_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, ignore=f1_ignore_cls,Ptype='Middle')
    end_pap1, end_pap_mean, end_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, ignore=f1_ignore_cls, Ptype='End')
    # 1.2 point-F1
    start_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls )
    mid_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 1.3. frame-wise results
    all_pred, all_gt = np.concatenate(list(pred_scores.values()), axis=0), np.concatenate(list(gt_targets.values()),axis=0)
    result_map = eval_perframe_map(cfg, all_gt, all_pred)
    result_prec, result_rec = eval_perframe_F1(cfg, all_gt, all_pred)
    result_acc = eval_perframe_acc(cfg, all_gt, all_pred)

    # 1.4 segment F1 score
    sess_pred, sess_gt = [], []
    for key in pred_scores.keys():
        output = pred_scores[key]
        single_pred = np.argmax(output, axis=-1)
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1,  bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25,  bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # global
    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': '-',
                  'mAP_fg': result_map['mean_AP_fg'] * 100, 'mAP_bg': result_map['mean_AP_bg'] * 100,
                  'mRec_fg': result_rec['mean_rec_fg'] * 100, 'mRec_bg': result_rec['mean_rec_bg'] * 100,
                  'mPrec_fg': result_prec['mean_prec_fg'] * 100, 'mPrec_bg': result_prec['mean_prec_bg'] * 100,
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_mid@1': mid_pap1 * 100,  'mPAP_mid@1-5': mid_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10,  'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': result_acc['accuracy'] * 100, 'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50,
                  'Edit': edit,
                  '||Extra|| ': ' || ',  # extra results
                  'mPAP_start@1': start_pap1 * 100, 'mPAP_start@1-5': start_pap_mean * 100,
                  'mPAP_end@1': end_pap1 * 100, 'mPAP_end@1-5': end_pap_mean * 100,
                  '||Point-F1|| ': ' || ',
                  'Start_1s(F1, prec, rec)': start_pF1[0], 's2s': start_pF1[1], 's3s': start_pF1[2], 's4s': start_pF1[3],
                  's5s': start_pF1[4], 'savg': start_pF1[5],
                  '|| ': ' || ',
                  'Mid_1s(F1, prec, rec)': mid_pF1[0], 'm2s': mid_pF1[1], 'm3s': mid_pF1[2], 'm4s': mid_pF1[3],
                  'm5s': mid_pF1[4], 'mavg': mid_pF1[5],
                  }

    df = pd.DataFrame([result_all])
    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head, float_format='%.1f')

    # 2. results w/ post process
    threshold = [1, 2, 3, 4, 5] # in secs
    for t in threshold:
        pp_size = t*cfg.DATA.FPS
        if pp_size %2 == 0:
            pp_size += 1
        postprocess_results(cfg, pred_scores, gt_targets, pp_size, f1_ignore_cls)


def postprocess_results_crosstask(cfg, pred_scores, gt_targets, pp_size, f1_ignore_cls):
    class_names = cfg.DATA.CLASS_NAMES
    sess_pred, sess_gt = [], []
    smoothed_pred_score = {}
    from scipy.ndimage import median_filter
    for key in pred_scores.keys():
        output = pred_scores[key]
        # post prcessing on prediction. this is for F1-based metric
        single_pred = np.argmax(output, axis=-1)
        single_pred_new = median_filter(single_pred, mode="nearest", size=pp_size)
        pad_start = np.array([single_pred_new[0]] * int(pp_size // 2))
        single_pred = np.concatenate((pad_start, single_pred_new[:-int(pp_size // 2)]))
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

        # post processing to smooth on probability, this is done only for map metric
        smoothed_output = np.zeros_like(output)
        for c in range(output.shape[1]):
            if c != cfg.DATA.IGNORE_INDEX:
                smoothed_output_new = median_filter(output[:, c], mode="nearest", size=pp_size)
                pad_output = np.array([smoothed_output_new[0]] * int(pp_size // 2))
                smoothed_output[:, c] = np.concatenate((pad_output, smoothed_output_new[:-int(pp_size // 2)]))
        smoothed_output = smoothed_output / smoothed_output.sum(1, keepdims=True)
        smoothed_pred_score[key] = smoothed_output

    # 1. point-mAP
    start_sm_pap1, start_sm_pap_mean, start_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets,
                                                                       Ptype='Start', ignore=f1_ignore_cls,
                                                                       data='crosstask')
    mid_sm_pap1, mid_sm_pap_mean, mid_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='Middle',
                                                                 ignore=f1_ignore_cls, data='crosstask')
    end_sm_pap1, end_sm_pap_mean, end_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='End',
                                                                 ignore=f1_ignore_cls, data='crosstask')

    # 1.2 point-F1
    start_pF1_sm = compute_point_F1(cfg, smoothed_pred_score, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1_sm = compute_point_F1(cfg, smoothed_pred_score, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 2. segment F1 score
    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1,  bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # global
    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': 'median' + '_s{}'.format(pp_size),
                  'mAP_fg': '-', 'mAP_bg': '-',
                  'mRec_fg': '-', 'mRec_bg': '-',
                  'mPrec_fg': '-', 'mPrec_bg': '-',
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_mid@1': mid_sm_pap1 * 100, 'mPAP_mid@1-5': mid_sm_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10, 'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': '-', 'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50,
                  'Edit': edit,
                  '||Extra|| ': ' || ',  # extra results
                  'mPAP_start@1': start_sm_pap1 * 100, 'mPAP_start@1-5': start_sm_pap_mean * 100,
                  'mPAP_end@1': end_sm_pap1 * 100, 'mPAP_end@1-5': end_sm_pap_mean * 100,
                  '||Point-F1|| ': ' || ',
                  'Start_1s(F1, prec, rec)': start_pF1_sm[0], 's2s': start_pF1_sm[1], 's3s': start_pF1_sm[2], 's4s': start_pF1_sm[3],
                  's5s': start_pF1_sm[4], 'savg': start_pF1_sm[5],
                  '|| ': ' || ',
                  'Mid_1s(F1, prec, rec)': mid_pF1_sm[0], 'm2s': mid_pF1_sm[1], 'm3s': mid_pF1_sm[2], 'm4s': mid_pF1_sm[3],
                  'm5s': mid_pF1_sm[4], 'mavg': mid_pF1_sm[5],
                  }
    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head,
              float_format='%.1f')


@compute_result_new.register('CrossTask_final')
def crosstask_results_final(cfg, pred_scores, gt_targets, logger):
    class_names = cfg.DATA.CLASS_NAMES
    # ## write prediction results
    # import os
    # output_path = os.path.splitext(cfg.MODEL.CHECKPOINT)[0] + '_{}_pred'.format(cfg.MODEL.LSTR.INFERENCE_MODE)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # for session in pred_scores:
    #     log_file = os.path.join(output_path, '{}.txt'.format(session))
    #     y = np.argmax(gt_targets[session], axis=-1)
    #     hat_y = np.argmax(pred_scores[session], axis=-1)  # changed to include bg
    #     # valid_ind = (y != cfg.DATA.IGNORE_INDEX) # remove y=0
    #     valid_ind = (y != -100)  # keep background
    #
    #     y = y[valid_ind]
    #     hat_y = hat_y[valid_ind]
    #     stamp = np.arange(len(valid_ind))[valid_ind]
    #
    #     out_str = ''
    #     for i in range(len(y)):
    #         out_str += '%-10s \t %-20s \t %-20s \n' % (str(stamp[i]), class_names[y[i]], class_names[hat_y[i]])
    #
    #     f_ptr = open(log_file, "w")
    #     f_ptr.write('%-10s \t %-20s \t %-20s \n' % ('Stamp', 'GT', 'Pred'))
    #     f_ptr.write(out_str)
    #     f_ptr.close()
    if cfg.DATA.IGNORE_INDEX > 0:
        f1_ignore_cls = list(set([0, cfg.DATA.IGNORE_INDEX]))
    else:
        f1_ignore_cls = [0]

    ## 1. [results w/o post process]
    # 1. point-mAP
    start_pap1, start_pap_mean, start_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Start', ignore=f1_ignore_cls, data='crosstask')
    mid_pap1, mid_pap_mean, mid_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Middle', ignore=f1_ignore_cls, data='crosstask')
    end_pap1, end_pap_mean, end_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='End', ignore=f1_ignore_cls, data='crosstask')

    # 1.2 point-F1
    start_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 3. frame-wise results
    all_pred, all_gt = np.concatenate(list(pred_scores.values()), axis=0), np.concatenate(list(gt_targets.values()), axis=0)
    result_map = eval_perframe_map(cfg, all_gt, all_pred)
    result_prec, result_rec = eval_perframe_F1(cfg, all_gt, all_pred)
    result_acc = eval_perframe_acc(cfg, all_gt, all_pred)

    # 1.4 segment F1 score
    sess_pred, sess_gt = [], []
    for key in pred_scores.keys():
        output = pred_scores[key]
        single_pred = np.argmax(output, axis=-1)
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # global
    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': '-',
                  'mAP_fg': result_map['mean_AP_fg'] * 100, 'mAP_bg': result_map['mean_AP_bg'] * 100,
                  'mRec_fg': result_rec['mean_rec_fg'] * 100, 'mRec_bg': result_rec['mean_rec_bg'] * 100,
                  'mPrec_fg': result_prec['mean_prec_fg'] * 100, 'mPrec_bg': result_prec['mean_prec_bg'] * 100,
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_mid@1': mid_pap1 * 100, 'mPAP_mid@1-5': mid_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10, 'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': result_acc['accuracy'] * 100, 'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50,
                  'Edit': edit,
                  '||Extra|| ': ' || ',  # extra results
                  'mPAP_start@1': start_pap1 * 100, 'mPAP_start@1-5': start_pap_mean * 100,
                  'mPAP_end@1': end_pap1 * 100, 'mPAP_end@1-5': end_pap_mean * 100,
                  '||Point-F1|| ': ' || ',
                  'Start_1s(F1, prec, rec)': start_pF1[0], 's2s': start_pF1[1], 's3s': start_pF1[2], 's4s': start_pF1[3],
                  's5s': start_pF1[4], 'savg': start_pF1[5],
                  '|| ': ' || ',
                  'Mid_1s(F1, prec, rec)': mid_pF1[0], 'm2s': mid_pF1[1], 'm3s': mid_pF1[2], 'm4s': mid_pF1[3],
                  'm5s': mid_pF1[4], 'mavg': mid_pF1[5],}
    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head,
              float_format='%.1f')

    # 2. results w/ post process
    threshold = [3, 5, 7, 9]  # in secs
    for t in threshold:
        pp_size = t * cfg.DATA.FPS
        if pp_size % 2 == 0:
            pp_size += 1
        postprocess_results_crosstask(cfg, pred_scores, gt_targets, pp_size, f1_ignore_cls)

####
def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0
    # np.zeros((scores.shape[0], scores.shape[1]))
    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls / len(classes)


def topk_recall_multiple_timesteps(preds, labels, k=5, classes=None):
    accs = np.array([topk_recall(preds[:, t, :], labels, k, classes)
                     for t in range(preds.shape[1])])
    return accs.reshape(1, -1)

def postprocess_results_ek100(cfg, pred_scores, gt_targets, pp_size, f1_ignore_cls):
    class_names = cfg.DATA.CLASS_NAMES
    sess_pred, sess_gt = [], []
    smoothed_pred_score = {}
    from scipy.ndimage import median_filter
    for key in pred_scores.keys():
        output = pred_scores[key]
        # post prcessing on prediction. this is for F1-based metric
        single_pred = np.argmax(output, axis=-1)
        single_pred_new = median_filter(single_pred, mode="nearest", size=pp_size)
        pad_start = np.array([single_pred_new[0]] * int(pp_size // 2))
        single_pred = np.concatenate((pad_start, single_pred_new[:-int(pp_size // 2)]))
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

        # post processing to smooth on probability, this is done only for map metric
        smoothed_output = np.zeros_like(output)
        for c in range(output.shape[1]):
            if c != cfg.DATA.IGNORE_INDEX:
                smoothed_output_new = median_filter(output[:, c], mode="nearest", size=pp_size)
                pad_output = np.array([smoothed_output_new[0]] * int(pp_size // 2))
                smoothed_output[:, c] = np.concatenate((pad_output, smoothed_output_new[:-int(pp_size // 2)]))
        smoothed_output = smoothed_output / smoothed_output.sum(1, keepdims=True)
        smoothed_pred_score[key] = smoothed_output

    # 1. point-mAP
    start_sm_pap1, start_sm_pap_mean, start_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets,
                                                                       Ptype='Start', ignore=f1_ignore_cls,
                                                                       data='ek100')
    mid_sm_pap1, mid_sm_pap_mean, mid_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='Middle',
                                                                 ignore=f1_ignore_cls, data='ek100')
    # end_sm_pap1, end_sm_pap_mean, end_sm_pap1_cls = evaluate_pAP(cfg, smoothed_pred_score, gt_targets, Ptype='End',
    #                                                              ignore=f1_ignore_cls, data='ek100')

    # 1.2 point-F1
    start_pF1_sm = compute_point_F1(cfg, smoothed_pred_score, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1_sm = compute_point_F1(cfg, smoothed_pred_score, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 2. segment F1 score
    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1,  bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # global
    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': 'median' + '_s{}'.format(pp_size),
                  'mAP_verb': '_', 'mAP_noun': '_', 'mAP_action': '_',
                  '|': '|',
                  'Rec5_verb': '_', 'Rec5_noun': '_', 'Rec5_action': '_',
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_start@1': start_sm_pap1 * 100, 'mPAP_start@1-5': start_sm_pap_mean * 100,
                  'mPAP_mid@1': mid_sm_pap1 * 100, 'mPAP_mid@1-5': mid_sm_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10, 'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': '-', 'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50,
                  'Edit': edit,
                  '||Point-F1|| ': ' || ',
                  'Start_1s(F1, prec, rec)': start_pF1_sm[0], 's2s': start_pF1_sm[1], 's3s': start_pF1_sm[2], 's4s': start_pF1_sm[3],
                  's5s': start_pF1_sm[4], 'savg': start_pF1_sm[5],
                  '|| ': ' || ',
                  'Mid_1s(F1, prec, rec)': mid_pF1_sm[0], 'm2s': mid_pF1_sm[1], 'm3s': mid_pF1_sm[2], 'm4s': mid_pF1_sm[3],
                  'm5s': mid_pF1_sm[4], 'mavg': mid_pF1_sm[5],
                  }
    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head,
              float_format='%.1f')


@compute_result_new.register('EK100_final_old')
def ek100_results_final_old(cfg, pred_scores, gt_targets, pred_scores_verb, gt_verb, pred_scores_noun, gt_noun, logger):
    class_names = cfg.DATA.CLASS_NAMES
    if cfg.DATA.IGNORE_INDEX > 0:
        f1_ignore_cls = list(set([0, cfg.DATA.IGNORE_INDEX]))
    else:
        f1_ignore_cls = [0]

    ## 1. [results w/o post process]
    # 1. point-mAP
    start_pap1, start_pap_mean, start_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Start', ignore=f1_ignore_cls, data='ek100')
    mid_pap1, mid_pap_mean, mid_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='Middle', ignore=f1_ignore_cls, data='ek100')
    #end_pap1, end_pap_mean, end_pap1_cls = evaluate_pAP(cfg, pred_scores, gt_targets, Ptype='End', ignore=f1_ignore_cls, data='crosstask')

    # 1.2 point-F1
    start_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls)
    mid_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls, dtype='Mid')

    # 3. frame-wise results
    map_result_act = eval_perframe(cfg, np.concatenate(list(gt_targets.values()), axis=0),
                                   np.concatenate(list(pred_scores.values()), axis=0))['mean_AP']

    map_result_noun = eval_perframe(cfg, np.concatenate(list(gt_noun.values()), axis=0),
                                    np.concatenate(list(pred_scores_noun.values()), axis=0),
                                    class_names = [str(i) for i in range(301)])['mean_AP']

    map_result_verb = eval_perframe(cfg, np.concatenate(list(gt_verb.values()), axis=0),
                                    np.concatenate(list(pred_scores_verb.values()), axis=0),
                                    class_names = [str(i) for i in range(98)])['mean_AP']

    oad_score, oad_noun_score, oad_verb_score = np.concatenate(list(pred_scores.values()), axis=0), \
                                                np.concatenate(list(pred_scores_noun.values()), axis=0), \
                                                np.concatenate(list(pred_scores_verb.values()), axis=0)
    oad_target, oad_noun_target, oad_verb_target = np.concatenate(list(gt_targets.values()), axis=0), \
                                                   np.concatenate(list(gt_noun.values()), axis=0), \
                                                   np.concatenate(list(gt_verb.values()), axis=0)

    action_labels, noun_labels, verb_labels = np.argmax(oad_target, axis=-1), \
                                              np.argmax(oad_noun_target, axis=-1), np.argmax(oad_verb_target, axis=-1)
    action_pred, oad_noun_pred, oad_verb_pred = oad_score.reshape(-1, 1, oad_score.shape[-1]), \
                                                oad_noun_score.reshape(-1, 1, oad_noun_score.shape[-1]), \
                                                oad_verb_score.reshape(-1, 1, oad_verb_score.shape[-1])
    valid_index = list(np.where(action_labels != 0))[0]
    overall_act_res = topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0][0]
    overall_noun_res = topk_recall_multiple_timesteps(oad_noun_pred[valid_index, ...], noun_labels[valid_index], k=5)[0][0]
    overall_verb_res = topk_recall_multiple_timesteps(oad_verb_pred[valid_index, ...], verb_labels[valid_index], k=5)[0][0]

    result_acc = eval_perframe_acc(cfg, oad_target, oad_score)

    # 1.4 segment F1 score
    sess_pred, sess_gt = [], []
    for key in pred_scores.keys():
        output = pred_scores[key]
        single_pred = np.argmax(output, axis=-1)
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

    p_f1_10, r_f1_10, b_f1_10 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.1, bg_class=f1_ignore_cls)
    p_f1_25, r_f1_25, b_f1_25 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.25, bg_class=f1_ignore_cls)
    p_f1_50, r_f1_50, b_f1_50 = overlap_f1_macro(sess_pred, sess_gt, len(class_names), overlap=0.5, bg_class=f1_ignore_cls)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    b_f1_10 = np.sum(b_f1_10) / (len(class_names) - len(f1_ignore_cls))
    b_f1_25 = np.sum(b_f1_25) / (len(class_names) - len(f1_ignore_cls))
    b_f1_50 = np.sum(b_f1_50) / (len(class_names) - len(f1_ignore_cls))

    p_f1_10 = np.sum(p_f1_10) / (len(class_names) - len(f1_ignore_cls))
    p_f1_25 = np.sum(p_f1_25) / (len(class_names) - len(f1_ignore_cls))
    p_f1_50 = np.sum(p_f1_50) / (len(class_names) - len(f1_ignore_cls))

    r_f1_10 = np.sum(r_f1_10) / (len(class_names) - len(f1_ignore_cls))
    r_f1_25 = np.sum(r_f1_25) / (len(class_names) - len(f1_ignore_cls))
    r_f1_50 = np.sum(r_f1_50) / (len(class_names) - len(f1_ignore_cls))

    # global
    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': '-',
                  'mAP_verb': map_result_verb * 100, 'mAP_noun': map_result_noun * 100,
                  'mAP_action': map_result_act * 100,
                  '|': '|',
                  'Rec5_verb': overall_verb_res * 100, 'Rec5_noun': overall_noun_res * 100,
                  'Rec5_action': overall_act_res * 100,
                  '[fram|point]': ' | ',  # separation from perframe to persegment
                  'mPAP_start@1': start_pap1 * 100, 'mPAP_start@1-5': start_pap_mean * 100,
                  'mPAP_mid@1': mid_pap1 * 100, 'mPAP_mid@1-5': mid_pap_mean * 100,
                  '[point|seg]': ' | ',  # separation from perframe to persegment
                  'mF1_10': b_f1_10, 'mF1_25': b_f1_25, 'mF1_50': b_f1_50,
                  'mRec_10': r_f1_10, 'mPrec_10': p_f1_10, 'mRec_25': r_f1_25, 'mPrec_25': p_f1_25,
                  'mRec_50': r_f1_50, 'mPrec_50': p_f1_50,
                  '||Global|| ': ' || ',  # separation from per calss to global
                  'Accuracy': result_acc['accuracy'] * 100, 'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50,
                  'Edit': edit,
                  '||Point-F1|| ': ' || ',
                  'Start_1s(F1, prec, rec)': start_pF1[0], 's2s': start_pF1[1], 's3s': start_pF1[2], 's4s': start_pF1[3],
                  's5s': start_pF1[4], 'savg': start_pF1[5],
                  '|| ': ' || ',
                  'Mid_1s(F1, prec, rec)': mid_pF1[0], 'm2s': mid_pF1[1], 'm3s': mid_pF1[2], 'm4s': mid_pF1[3],
                  'm5s': mid_pF1[4], 'mavg': mid_pF1[5],}
    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head,
              float_format='%.1f')

    # 2. results w/ post process
    threshold = [1, 2, 3]  # in secs
    for t in threshold:
        pp_size = t * cfg.DATA.FPS
        if pp_size % 2 == 0:
            pp_size += 1
        postprocess_results_ek100(cfg, pred_scores, gt_targets, pp_size, f1_ignore_cls)


@compute_result_new.register('EK100_final')
def ek100_results_final(cfg, pred_scores, gt_targets, pred_scores_verb, gt_verb, pred_scores_noun, gt_noun, logger):
    class_names = cfg.DATA.CLASS_NAMES
    if cfg.DATA.IGNORE_INDEX > 0:
        f1_ignore_cls = list(set([0, cfg.DATA.IGNORE_INDEX]))
    else:
        f1_ignore_cls = [0]

    ## 1. [results w/o post process]
    # point-F1
    start_pF1 = compute_point_F1(cfg, pred_scores, gt_targets, bg_class=f1_ignore_cls, dist_ths = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

    # 3. frame-wise results
    map_result_act = eval_perframe(cfg, np.concatenate(list(gt_targets.values()), axis=0),
                                   np.concatenate(list(pred_scores.values()), axis=0))['mean_AP']

    map_result_noun = eval_perframe(cfg, np.concatenate(list(gt_noun.values()), axis=0),
                                    np.concatenate(list(pred_scores_noun.values()), axis=0),
                                    class_names = [str(i) for i in range(301)])['mean_AP']

    map_result_verb = eval_perframe(cfg, np.concatenate(list(gt_verb.values()), axis=0),
                                    np.concatenate(list(pred_scores_verb.values()), axis=0),
                                    class_names = [str(i) for i in range(98)])['mean_AP']

    oad_score, oad_noun_score, oad_verb_score = np.concatenate(list(pred_scores.values()), axis=0), \
                                                np.concatenate(list(pred_scores_noun.values()), axis=0), \
                                                np.concatenate(list(pred_scores_verb.values()), axis=0)
    oad_target, oad_noun_target, oad_verb_target = np.concatenate(list(gt_targets.values()), axis=0), \
                                                   np.concatenate(list(gt_noun.values()), axis=0), \
                                                   np.concatenate(list(gt_verb.values()), axis=0)

    action_labels, noun_labels, verb_labels = np.argmax(oad_target, axis=-1), \
                                              np.argmax(oad_noun_target, axis=-1), np.argmax(oad_verb_target, axis=-1)
    action_pred, oad_noun_pred, oad_verb_pred = oad_score.reshape(-1, 1, oad_score.shape[-1]), \
                                                oad_noun_score.reshape(-1, 1, oad_noun_score.shape[-1]), \
                                                oad_verb_score.reshape(-1, 1, oad_verb_score.shape[-1])
    valid_index = list(np.where(action_labels != 0))[0]
    overall_act_res = topk_recall_multiple_timesteps(action_pred[valid_index, ...], action_labels[valid_index], k=5)[0][0]
    overall_noun_res = topk_recall_multiple_timesteps(oad_noun_pred[valid_index, ...], noun_labels[valid_index], k=5)[0][0]
    overall_verb_res = topk_recall_multiple_timesteps(oad_verb_pred[valid_index, ...], verb_labels[valid_index], k=5)[0][0]

    # 1.4 segment F1 score
    sess_pred, sess_gt = [], []
    for key in pred_scores.keys():
        output = pred_scores[key]
        single_pred = np.argmax(output, axis=-1)
        single_gt = np.argmax(gt_targets[key], axis=-1)
        sess_pred.append(single_pred)
        sess_gt.append(single_gt)

    edit = edit_score(sess_pred, sess_gt, bg_class=f1_ignore_cls)

    gp_f1_10, gr_f1_10, gb_f1_10 = overlap_f1(sess_pred, sess_gt, overlap=0.1, bg_class=f1_ignore_cls)
    gp_f1_25, gr_f1_25, gb_f1_25 = overlap_f1(sess_pred, sess_gt, overlap=0.25, bg_class=f1_ignore_cls)
    gp_f1_50, gr_f1_50, gb_f1_50 = overlap_f1(sess_pred, sess_gt, overlap=0.5, bg_class=f1_ignore_cls)


    result_all = {'method': cfg.MODEL.CHECKPOINT.split('/')[-2],
                  'epoch': cfg.MODEL.CHECKPOINT.split('/')[-1].split('.')[0],
                  'mode': cfg.MODEL.LSTR.INFERENCE_MODE,
                  'postproc': '-',
                  'mAP_verb': map_result_verb * 100, 'mAP_noun': map_result_noun * 100,
                  'mAP_action': map_result_act * 100,
                  '|': '|',
                  'Rec5_verb': overall_verb_res * 100, 'Rec5_noun': overall_noun_res * 100,
                  'Rec5_action': overall_act_res * 100,
                  '[fram|seg]': ' | ',  # separation from perframe to persegment
                  'F1_10': gb_f1_10, 'F1_25': gb_f1_25, 'F1_50': gb_f1_50,
                  'Edit': edit,
                  '||Point-F1|| ': ' || ',
                  'Start_0.5s(F1, prec, rec)': start_pF1[0], '1s': start_pF1[1], '2s': start_pF1[2], '3s': start_pF1[3],
                  '4s': start_pF1[4], '5s': start_pF1[5], 'avg': start_pF1[6],}
    df = pd.DataFrame([result_all])

    head = True
    if os.path.exists('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME)):
        head = False
    df.to_csv('{}_summary_final.csv'.format(cfg.DATA.DATA_NAME), mode='a', index=False, header=head,
              float_format='%.1f')

