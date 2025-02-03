import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import time
import librosa
from tqdm import tqdm
DATASET_PATH = "/workspace/nas-dataset/jis/voicebank_demand/"



class CustomDataset(Dataset):
    def __init__(self, train=True,target_len=16000, transform=None, fixed=False, bigfile=False):
        if train:
            self.clean_path = os.path.join(DATASET_PATH, 'clean_trainset_28spk_wav')
            self.noisy_path = os.path.join(DATASET_PATH, 'noisy_trainset_28spk_wav')

        else:
            self.clean_path = os.path.join(DATASET_PATH, 'clean_testset_wav')
            self.noisy_path = os.path.join(DATASET_PATH, 'noisy_testset_wav')

                
        self.file_list = os.listdir(self.clean_path)
        self.file_list.sort()
        
        self.target_len = target_len
        self.transform = transform
        self.fixed = fixed
        
        
        self.clean_list = []
        self.noisy_list = []
        self.x1_list = []
        if bigfile:
            self.clean_list, self.noisy_list, self.x1_list = torch.load(f"{'train' if train else 'test'}_file_list.pt")
        else:
            for file in tqdm(self.file_list):
                clean, sr1 = librosa.load(os.path.join(self.clean_path, file), sr=16000,res_type='kaiser_best')
                noisy, sr2 = librosa.load(os.path.join(self.noisy_path, file), sr=16000,res_type='kaiser_best')

                self.clean_list.append(clean)
                self.noisy_list.append(noisy)            
    


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        clean = torch.FloatTensor(self.clean_list[index]).unsqueeze(0)
        noisy = torch.FloatTensor(self.noisy_list[index]).unsqueeze(0)
        
        
        assert clean.shape == noisy.shape
        

        pad = max(self.target_len - clean.shape[-1], 0)
        if (self.target_len - clean.shape[-1]) == 0:
            pass
        elif pad > 0:
            clean = torch.nn.functional.pad(clean, (pad//2, pad//2+(pad%2)), mode='constant')
            noisy = torch.nn.functional.pad(noisy, (pad//2, pad//2+(pad%2)), mode='constant')
        else:
            if self.fixed:
                start_point = hash(self.file_list[index]) % (clean.shape[-1] - self.target_len)
            else:
                start_point = torch.randint(0, clean.shape[-1] - self.target_len, (1,)).item()
            start_point = torch.randint(0, clean.shape[-1] - self.target_len, (1,)).item()
            clean = clean[:, start_point:start_point + self.target_len]
            noisy = noisy[:, start_point:start_point + self.target_len]
        
        wav = (clean, noisy)


        return wav
