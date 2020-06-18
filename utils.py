import argparse
import logging

import cv2 as cv
import librosa
import numpy as np
import torch,os
import tqdm
from config import sample_rate
from world_utils import *
import multiprocessing
from waveglow_vocoder import WaveGlowVocoder

WV = WaveGlowVocoder()

print ('using sample rate of {}'.format(sample_rate))

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, acc, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'metric_fc': metric_fc,
             'optimizer': optimizer}

    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Speaker Embeddings')
    # Training config
    parser.add_argument('--epochs', default=1000, type=int, help='Number of maximum epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Init learning rate')
    parser.add_argument('--l2', default=1e-6, type=float, help='weight decay (L2)')
    parser.add_argument('--batch-size', default=16, type=int, help='Batch size')
    parser.add_argument('--num-workers', default=10, type=int, help='Number of workers to generate minibatch')
    # optimizer
    parser.add_argument('--margin-m', type=float, default=0.2, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=10.0, help='feature scale s')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


# [-0.5, 0.5]
def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt


# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
from pathlib import Path
import json
    
save_folder = './extracted_features'
wavs_folder = '/resources/wavs/Speaker-Embeddings/data/vox1'
# wavs_folder = '/resources/wavs/Speaker-Embeddings/data/vox1/test/wav/id10270/5sJomL_D0_g'
def get_new_path(x):
    split = x.split('/')
    return os.path.join(save_folder,split[-3]+'_'+split[-2]+'_'+split[-1]+'.npy')

def extract_save(x):
    feature = extract_feature(x)
    new_file_path = get_new_path(x)
    np.save(new_file_path,feature)
    return (x,new_file_path)



def extract_feature(input_file, feature='world', dim=48, cmvn=False, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    if feature == 'pre_world':
            if input_file.endswith('.wav'):
                pre_processed_sample_path =get_new_path(input_file)
                l = np.load(pre_processed_sample_path)
            else:
                l = np.load(input_file)
            return l
    if feature == 'glow':
        y, sr = librosa.load(input_file, sr=22050)
        y = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)
        mel = WV.wav2mel(y).cpu().numpy()[0]
        return mel
        
    y, sr = librosa.load(input_file, sr=sample_rate)
    yt, _ = librosa.effects.trim(y, top_db=20)
    yt = normalize(yt)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim, n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)
        print (feat.shape)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=26, n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(yt, hop_length=st, frame_length=ws)
    elif feature == 'world':
        f0, timeaxis, sp, ap = world_decompose(wav=y, fs=sr, frame_period= 5.0)
        feat = world_encode_spectral_envelop(sp = sp, fs = sr, dim = dim).T

    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


def build_LFR_features_numpy(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    inputs = torch.Tensor(inputs)
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(torch.cat(list(inputs[i * n:i * n + m])))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = torch.cat(list(inputs[i * n:]))
            for _ in range(num_padding):
                frame = torch.cat((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return torch.stack(LFR_inputs)


def theta_dist():
    from test import image_name
    img = cv.imread(image_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img / 255.
    return img


if __name__ == "__main__":
    os.makedirs(save_folder,exist_ok=True)
    p = multiprocessing.Pool(6)

    wavs = [ str(path) for path in Path(wavs_folder).rglob('*.wav')]
    
    mapping = list(tqdm.tqdm(p.imap_unordered(extract_save,wavs),total=len(wavs)))
    
    mapping_dict = {}
    
    for x,y in mapping:
        mapping_dict[x] = y


    with open('mapping.json', 'w') as file:
         file.write(json.dumps(mapping_dict)) 