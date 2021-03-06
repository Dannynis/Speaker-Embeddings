import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
n_mels = 48  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT (ms)
hidden_size = 512
embedding_dim = 512
cmvn = True  # apply CMVN on feature
LFR_m = 4  # change to 4 if use LFR
LFR_n = 3

# Reference encoder
# ref_enc_filters = [32, 32, 64, 64, 128, 128]
ref_enc_filters = [64, 64, 128, 128, 256, 256]

# Style token layer
token_num = 1024
token_emb_size = 512
num_heads = 8

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 500  # print training/validation stats  every __ batches
print_freq_test = 20000
checkpoint = None  # path to checkpoint, None if none
sample_rate = 22050  # vox1

# Data parameters
num_train = 147642
num_valid = 1000

num_classes = 1211

DATA_DIR = '/resources/wavs/Speaker-Embeddings/'
vox1_folder = '/resources/wavs/Speaker-Embeddings/data/vox1'
dev_wav_folder = os.path.join(vox1_folder, 'dev/wav')
test_wav_folder = os.path.join(vox1_folder, 'test/wav')
data_file = '/resources/wavs/Speaker-Embeddings/data/vox1.pickle'

##Im the king
