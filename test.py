import os
import random

import torch
from tqdm import tqdm
import time
import config as hp
from models.embedder import GST
from utils import extract_feature
test_file = 'data/test_pairs.txt'


def gen_test_pairs():
    num_tests = 6000

    num_same = int(num_tests / 2)
    num_not_same = num_tests - num_same

    out_lines = []

    for _ in tqdm(range(num_same)):
        dirs = [d for d in os.listdir(hp.test_wav_folder) if os.path.isdir(os.path.join(hp.test_wav_folder, d))]
        folder = random.choice(dirs)
        folder = os.path.join(hp.test_wav_folder, folder)
        file_list = []
        for root, dir, files in os.walk(folder):
            files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
            file_list += files
        pair = random.sample(file_list, 2)
        out_lines.append('{} {} {}\n'.format(pair[0], pair[1], 1))

    for _ in tqdm(range(num_not_same)):
        dirs = [d for d in os.listdir(hp.test_wav_folder) if os.path.isdir(os.path.join(hp.test_wav_folder, d))]
        folders = [os.path.join(hp.test_wav_folder, folder) for folder in random.sample(dirs, 2)]
        file_list_0 = []
        for root, dir, files in os.walk(folders[0]):
            files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
            file_list_0 += files
        file_0 = random.choice(file_list_0)
        file_list_1 = []
        for root, dir, files in os.walk(folders[1]):
            files = [os.path.join(root, file) for file in files if file.endswith('.wav')]
            file_list_1 += files
        file_1 = random.choice(file_list_1)
        out_lines.append('{} {} {}\n'.format(file_0, file_1, 0))

    with open(test_file, 'w') as file:
        file.writelines(out_lines)


def evaluate(model):
    with open(test_file, 'r') as file:
        lines = file.readlines()

    angles = []

    start = time.time()
    with torch.no_grad():
        for line in tqdm(lines):
            tokens = line.split()
            file0 = tokens[0]
            feature0  = extract_feature(input_file=file0, feature='fbank', dim=hp.n_mels, cmvn=True)
            file1 = tokens[1]
            feature1 = extract_feature(input_file=file1, feature='fbank', dim=hp.n_mels, cmvn=True)
            imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
            imgs[0] = img0
            imgs[1] = img1

            output = model(imgs)

            feature0 = output[0].cpu().numpy()
            feature1 = output[1].cpu().numpy()
            x0 = feature0 / np.linalg.norm(feature0)
            x1 = feature1 / np.linalg.norm(feature1)
            cosine = np.dot(x0, x1)
            theta = math.acos(cosine)
            theta = theta * 180 / math.pi
            is_same = tokens[2]
            angles.append('{} {}\n'.format(theta, is_same))

    elapsed_time = time.time() - start
    print('elapsed time(sec) per image: {}'.format(elapsed_time / (6000 * 2)))

    with open('data/angles.txt', 'w') as file:
        file.writelines(angles)


if __name__ == "__main__":
    checkpoint = 'speaker-embeddings.pt'
    print('loading model: {}...'.format(checkpoint))
    model = GST()
    model.load_state_dict(torch.load(checkpoint))
    model = model.to(hp.device)
    model.eval()
