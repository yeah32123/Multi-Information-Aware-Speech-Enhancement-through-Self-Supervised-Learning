from pathlib import Path
import os
import time
import pickle
import warnings
import gc
import copy

import noise_addition_utils

from metrics import AudioMetrics
from metrics import AudioMetrics2

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from pypesq import pesq
from IPython.display import clear_output
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.simplefilter ( "ignore")


from pesq import pesq
from scipy import interpolate
from loss_utils import mse_loss, stftm_loss

from DCUnet import DCUnet10, DCUnet10_cTSTM, DCUnet10_rTSTM

from network import DCUnet20

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'      # 使用 GPU 3 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random

noise_class = "white"
training_type = "IDDP"
Network = "DCUnet10_cTSTM"
loss = "x1_L2"
shuffle = "no shuffle"

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 
# N_FFT = 1022
# HOP_LENGTH = 256

# max_len = 65280
max_len = 165000
istft_len = 164352


# Load white noise
if noise_class == "white":
    TRAIN_INPUT_DIR = Path('/home/maoyj/projects/MIA-SE/Noise2Noise-audio_denoising_without_clean_training_data/Datasets/noisy_trainset_28spk_wav')
    TRAIN_TARGET_DIR = Path('/home/maoyj/projects/MIA-SE/Noise2Noise-audio_denoising_without_clean_training_data/Datasets/clean_trainset_28spk_wav')

    TEST_NOISY_DIR = Path('/home/maoyj/projects/MIA-SE/Noise2Noise-audio_denoising_without_clean_training_data/Datasets/noisy_testset_wav')
    TEST_CLEAN_DIR = Path('/home/maoyj/projects/MIA-SE/Noise2Noise-audio_denoising_without_clean_training_data/Datasets/clean_testset_wav')

# Load urbansound8K noise
else:
    TRAIN_INPUT_DIR = Path('/home/maoyj/projects/Denoise/wsz_Data/train/noisy')
    TRAIN_TARGET_DIR = Path('/home/maoyj/projects/Denoise/wsz_Data/train/clean')

    TEST_NOISY_DIR = Path('/home/maoyj/projects/Denoise/wsz_Data/valid/noisy')
    TEST_CLEAN_DIR = Path('/home/maoyj/projects/Denoise/wsz_Data/valid/clean')
    # TRAIN_INPUT_DIR = Path('/home/maoyj/projects/Noise2Noise-audio_denoising_without_clean_training_data/Datasets_US/US_Class' + str(noise_class) + '_Train_Input')
    # TRAIN_TARGET_DIR = Path('/home/maoyj/projects/Noise2Noise-audio_denoising_without_clean_training_data/Datasets_US/US_Class' + str(noise_class) + '_Train_Output')

    # TEST_NOISY_DIR = Path('/home/maoyj/projects/Noise2Noise-audio_denoising_without_clean_training_data/Datasets_US/US_Class' + str(noise_class) + '_Test_Input')
    # TEST_CLEAN_DIR = Path('/home/maoyj/projects/Noise2Noise-audio_denoising_without_clean_training_data/Datasets_US/clean_testset_wav')

basepath = str(noise_class)+"_"+training_type+"_"+Network+loss+"_"+loss+shuffle
os.makedirs(basepath,exist_ok=True)
os.makedirs(basepath+"/Weights",exist_ok=True)
os.makedirs(basepath+"/Samples",exist_ok=True)

np.random.seed(999)
torch.manual_seed(999)

# If running on Cuda set these 2 for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_on_gpu = torch.cuda.is_available()
torch.cuda.set_device(2)


if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
       
DEVICE = torch.device('cuda', 1 if train_on_gpu else 'cpu')

torchaudio.set_audio_backend("sox_io")
# print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))

def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old  = np.linspace(0, duration, original.shape[0])
        time_new  = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


class SpeechDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """
    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        
        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.len_ = len(self.noisy_files)
        # print(self.len_)
        
        # fixed len
        # self.max_len = 165000
        self.max_len = max_len

    
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform
  
    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])
        
        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        
        # Short-time Fourier transform
        # print("noisy", x_noisy.shape)
        # 这里会有警告
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True)
        # print("stft", x_noisy_stft.shape)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True)
        
        return x_noisy_stft, x_clean_stft
        
    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        
        # output = np.zeros((1, self.max_len), dtype='float32')
        # if current_len <= self.max_len:
        #     n_repeat = int(np.ceil(float(self.max_len) / float(current_len)))
        #     waveform = np.tile(waveform, n_repeat)
        # output[0, :self.max_len] = waveform[0, :self.max_len]
        # # else:
        # #     noise_onset = rs.randint(0, current_len - self.max_len, size=1)[0]
        # #     noise_offset = noise_onset + self.max_len
        # #     output[0, :self.max_len] = waveform[0, noise_onset:noise_offset]
        # output = torch.from_numpy(output)

        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        # output = resample(output, 44100, SAMPLE_RATE)
        output = torch.from_numpy(output)
        return output


time_loss = mse_loss()
freq_loss = stftm_loss()


def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False)
    for j, seq_len in enumerate(nframes):
        loss_mask.data[j, :, 0:seq_len] += 1.0   # loss_mask.shape: torch.Size([2, 1, 32512])
    return loss_mask

def wsdr_fn(x, y_pred, y_true, eps=1e-8):
    # to time-domain waveform
    # y_true_ = torch.squeeze(y_true_, 1)
    # y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    # x_ = torch.squeeze(x_, 1)
    # x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)

    y_pred = y_pred.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
 
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

def l2(x, y_true):
    y_true = y_true.flatten(1)
    x = x.flatten(1)
    return torch.sum(torch.norm((x-y_true), p=2, dim=1))
    
# 损失函数
def wsdr_tf(x, y_pred, y_true):
    # x:二次加噪以后的输入
    # y_pred:输入x以后的网络输出
    # y_true:二次加噪之前网络的输出
    if(y_true.shape[0] == 2):
            nframes = [y_true.shape[1],y_true.shape[1]]   # nframes: [32512, 32512]
    else:
            nframes = [y_true.shape[1]]
    print("nframes", nframes)
    print("y_true", y_true.shape)
    loss_mask = compLossMask(y_true.unsqueeze(1), nframes)   
    loss_mask = loss_mask.float().to(DEVICE)
    print("loss_mask", loss_mask.shape)
    print("y_pred.unsqueeze(1)", y_pred.unsqueeze(1).shape)
    print("y_true.unsqueeze(1)", y_true.unsqueeze(1).shape)
    loss_time = time_loss(y_pred.unsqueeze(1), y_true.unsqueeze(1), loss_mask)
    loss_freq = freq_loss(y_pred.unsqueeze(1), y_true.unsqueeze(1), loss_mask)
    loss1 = (0.8 * loss_time + 0.2 * loss_freq)/600
    loss = loss1 + wsdr_fn(x, y_pred, y_true) + 0.001*l2(x, y_true)

    return loss

wonky_samples = []

def getMetricsonLoader(loader, net, use_net=True):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB","PESQ-NB","SNR","SSNR","STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i+1)%10==0:
            end_str = "\n"
        else:
            end_str = ","
        #print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            
            noisy = data[0]
            clean = data[1]
            if use_net: # Forward of net returns the istft version
                x_est = net(noisy.to(torch.float32).to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE)
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                x_est_np = torch.istft(torch.squeeze(noisy, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
            x_clean_np = torch.istft(torch.squeeze(clean, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
            
        
            metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)
            
            ref_wb = resample(x_clean_np, 48000, 16000)
            deg_wb = resample(x_est_np, 48000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')
            
            ref_nb = resample(x_clean_np, 48000, 8000)
            deg_nb = resample(x_est_np, 48000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            #print(new_scores)
            #print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] =  np.mean(overall_metrics[i])
        temp["STD"]  =  np.std(overall_metrics[i])
        temp["Min"]  =  min(overall_metrics[i])
        temp["Max"]  =  max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data",addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    return results

def addnoise(original,snr):
    # print(len(original))
    N = np.random.randn(len(original)).astype(np.float32)
    numerator = sum(np.square(original.astype(np.float32)))
    denominator = sum(np.square(N))
    factor = 10**(snr/10.0)
    K = (numerator/(factor*denominator))**0.5
    noise = original + K*N
    # plt.subplot(3,1,1)
    # plt.plot(original)
    # plt.subplot(3,1,2)
    # plt.plot(N)
    # plt.subplot(3,1,3)
    # plt.plot(noise)
    # plt.savefig('debug.png')
    # plt.close()
    return noise

def addnoise_Gauss(original,snr):
    # print(len(original))
    current_len = len(original)
    zero_len = (original != 0).argmax(axis=0)
    signal_len = current_len - zero_len
    N = np.random.randn(signal_len).astype(np.float32)
    numerator = sum(np.square(original.astype(np.float32)))
    denominator = sum(np.square(N))
    factor = 10**(snr/10.0)
    K = (numerator/(factor*denominator))**0.5
    noise = K*N
    noise_add = np.zeros((1, current_len), dtype='float32')
    noise_add[0, -signal_len:] = noise
    noisy = original + noise_add
    # plt.subplot(4,1,1)
    # plt.plot(original)
    # plt.subplot(4,1,2)
    # plt.plot(N)
    # plt.subplot(4,1,3)
    # plt.plot(noise_add[0,:])
    # plt.subplot(4,1,4)
    # plt.plot(noisy[0,:])
    # plt.savefig('debug.png')
    # plt.close()
    return noisy

# 噪声信号文件夹
Noise_dir = "/home/maoyj/projects/Denoise/wsz_Data/train/noise"


def allWavFiles(directory):
    result = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            result.append(os.path.join(directory, filename))
    return result

rs = np.random.RandomState(0)
def addnoise_Dregon(original, snr):

    current_len = len(original)
    zero_len = (original != 0).argmax(axis=0)
    signal_len = current_len - zero_len
    
    try:
        # 从文件夹中读取wav噪声
        possible_noises = allWavFiles(Noise_dir)
        total_noise = len(possible_noises)
        samples = np.random.choice(total_noise, 1, replace=False)
        s = samples[0]
        noisefile = possible_noises[s]
        noise_src_file, _ = torchaudio.load(noisefile)
        noise_src_file = noise_src_file.numpy()
        noise_src_file = np.reshape(noise_src_file, -1)
        # noise = np.zeros((1, max_len), dtype='float32')
        # padding or cutting, 使噪声信号和输入信号长度相同
        if len(noise_src_file) <= signal_len:
            n_repeat = int(np.ceil(float(signal_len) / float(len(noise_src_file))))
            noise_src_file_ex = np.tile(noise_src_file, n_repeat)
            noise = noise_src_file_ex[0 : signal_len]
        else:
            noise_onset = rs.randint(0, len(noise_src_file) - signal_len, size=1)[0]
            noise_offset = noise_onset + signal_len 
            noise = noise_src_file[noise_onset : noise_offset]
    except:
        print('the noise file can not found')

    # 原始信号强度
    P_original = np.sum(np.abs(original) ** 2)
    # 原始噪声强度
    P_d = np.sum(np.abs(noise) ** 2)
    # 所需要的噪声强度
    P_noise = P_original / 10 ** (snr /10)
    # 所需要产生的噪声强度
    noise = np.sqrt(P_noise / P_d) * noise

    noise_add = np.zeros((1, current_len), dtype='float32')
    noise_add[0, -signal_len:] = noise
    # 混合的噪声信号
    noisy = original + noise_add


    return noisy

def addNoise_shuffle(pred_x1_np, noisy_x1):

    noisy_x1_np = torch.istft(torch.squeeze(noisy_x1, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
    noise = noisy_x1_np - pred_x1_np
    # 生成K个序列随机shuffle
    K = 10
    L = len(pred_x1_np) // K
    s = [x for x in range(K)]
    random.shuffle(s)
    noise_add = []
    for i in range(K):
        noise_add.append(noise[s[i]*L:(s[i]+1)*L])
    noise_add = np.hstack(noise_add)
    noise_add = noise_add - np.mean(noise_add)
    noisy_x2_np = pred_x1_np + noise_add

    return noisy_x2_np

# # 采样
# pred_x1_np = np.ones([2,10], dtype = float)
# noisy_x2 = np.zeros(pred_x1_np.shape)
# for i in range(len(pred_x1_np)):
#     # print(i)
#     noisy_x2[i] = addnoise(pred_x1_np[i],snr=np.random.randint(0,10))
# print(noisy_x2.shape)

mse_loss = nn.MSELoss(reduce=True, size_average=True)

def train_epoch(net, train_loader, loss_fn, optimizer, epoch):
    net.train()
    train_ep_loss = 0.
    counter = 0
    for idx, (noisy_x, _) in enumerate(train_loader):
        
        # noisy_x1, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)#dataloader输出是stft尺寸，[2,1,1537,215,2]
        noisy_x1 = noisy_x.to(DEVICE) #dataloader输出是stft尺寸，[2,1,1537,215,2]
        noisy_x1 = noisy_x1.to(torch.float32)
        noisy_x1_istft = torch.istft(torch.squeeze(noisy_x1, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
        # zero  gradients
        net.zero_grad()

        # get the output1 from the model
        # print('noisy_x1.shape', noisy_x1.shape) #输入是[2，1，1537，215，2]
        with torch.no_grad():
            pred_x1 = net(noisy_x1, n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE) #模型输出是istft的尺寸，[2,164352]
        # print('pred_x1_stft.shape', pred_x1_stft.shape)

        # pred_x1 = torch.istft(pred_x1_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
        # print('pred_x1.shape', pred_x1.shape)
        pred_x1_np = pred_x1.cpu().numpy()
        # pred_x1_np = pred_x1.detach().cpu().numpy()
        # 通过从已知的噪声分布采样一个新的噪声以对去噪的图像重新加噪
        noisy_x2 = np.zeros(pred_x1_np.shape, dtype='float32')
        for i in range(len(pred_x1_np)):
            # noisy_x2[i] = addnoise_Gauss(pred_x1_np[i],snr=np.random.randint(0,10))
            noisy_x2[i] = addnoise_Dregon(pred_x1_np[i],snr=np.random.randint(0,10))
            # noisy_x2[i] = addNoise_shuffle(pred_x1_np[i], noisy_x[i])
            
        # print('noisy_x2.shape', noisy_x2.shape)
        noisy_x2_stft = torch.stft(input=torch.from_numpy(np.array(noisy_x2)), n_fft=N_FFT,  #STFT是[2, 1537，215，2]
                                     hop_length=HOP_LENGTH, normalized=True)
        # print('noisy_x2_stft.shape', noisy_x2_stft.shape)
        # print('noisy_x2_stft.unsqueeze(1).shape', noisy_x2_stft.unsqueeze(1).shape)
        noisy_x2_stft = noisy_x2_stft.to(torch.float32)
        pred_x2 = net(noisy_x2_stft.unsqueeze(1).to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE)
        print('pred_x2', pred_x2.shape)
        noisy_x2 = torch.from_numpy(noisy_x2).to(DEVICE)
        # calulate loss
        loss = loss_fn(noisy_x1_istft, pred_x2, pred_x1.detach())

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx+1) % 10 == 0:
            print(
                'Train: [{0}][{1}/{2}]    '
                'Loss: {3}'.format(
                epoch + 1,
                idx + 1,
                len(train_loader),
                loss
                )
            )
        train_ep_loss += loss.item() 
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss

# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    '''
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item() 
        
        counter += 1

    test_ep_loss /= counter
    '''
    
    #print("Actual compute done...testing now")
    
    testmet = getMetricsonLoader(test_loader,net,use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss, testmet


def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    
    train_losses = []
    test_losses = []

    # for e in tqdm(range(epochs)):
    for e in range(epochs):
        # first evaluating for comparison
        
        if e == 0 and training_type=="Noise2Clean":
            print("Pre-training evaluation")
            #with torch.no_grad():
            #    test_loss,testmet = test_epoch(net, test_loader, loss_fn,use_net=False)
            #print("Had to load model.. checking if deets match")
            testmet = getMetricsonLoader(test_loader,net,False)    # again, modified cuz im loading
            #test_losses.append(test_loss)
            #print("Loss before training:{:.6f}".format(test_loss))
        
            with open(basepath + "/results.txt","w+") as f:
                f.write("Initial : \n")
                f.write(str(testmet))
                f.write("\n")
        
        print('--------------------Epoch:{}--------------------'.format(e+1))
        train_loss = train_epoch(net, train_loader, loss_fn, optimizer, e)
        test_loss = 0
        scheduler.step()
        print("Saving model....")
        
        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn,use_net=True)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        #print("skipping testing cuz peak autism idk")
        
        with open(basepath + "/results.txt","a") as f:
            f.write("Epoch :"+str(e+1) + "\n" + str(testmet))
            f.write("\n")
        
        print("OPed to txt")
        
        torch.save(net.state_dict(), basepath +'/Weights/dc10_model_'+str(e+1)+'.pth')
        torch.save(optimizer.state_dict(), basepath+'/Weights/dc10_opt_'+str(e+1)+'.pth')
        
        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        #print("Epoch: {}/{}...".format(e+1, epochs),
        #              "Loss: {:.6f}...".format(train_loss),
        #              "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss


if __name__ == '__main__':


    gc.collect()
    torch.cuda.empty_cache()
    dcunet10 = DCUnet10_cTSTM(N_FFT, HOP_LENGTH, DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(dcunet10.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss_fn = wsdr_tf
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 网络参数数量
    get_parameter_number(dcunet10)

    # 加载数据
    train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
    train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))

    test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
    test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

    print("No. of Training files:",len(train_input_files))
    print("No. of Testing files:",len(test_noisy_files))

    test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
    train_dataset = SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # # For testing purpose
    # test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)
    epochs = 2
    # specify paths and uncomment to resume training from a given point
    # path_to_model = "/home/maoyj/projects/Noise2Noise-audio_denoising_without_clean_training_data/white_IDDP_DCUnet10_cTSTMx1_L2_x1_L2no shuffle/Weights/dc10_model_16.pth"
    # path_to_opt = "/home/maoyj/projects/Noise2Noise-audio_denoising_without_clean_training_data/white_IDDP_DCUnet10_cTSTMx1_L2_x1_L2no shuffle/Weights/dc10_opt_16.pth"
    # model_checkpoint = torch.load(path_to_model)
    # opt_checkpoint = torch.load(path_to_opt)
    # dcunet10.load_state_dict(model_checkpoint)
    # optimizer.load_state_dict(opt_checkpoint)
    train_losses, test_losses = train(dcunet10, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs)
