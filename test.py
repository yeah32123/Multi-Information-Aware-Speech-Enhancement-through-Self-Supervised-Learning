import torch
from DCUnet10_TSTM.DCUnet import DCUnet10, DCUnet10_cTSTM, DCUnet10_rTSTM

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio
from metrics import AudioMetrics
from metrics import AudioMetrics2
import matplotlib.pyplot as plt

import noise_addition_utils
import numpy as np
from pesq import pesq
from scipy import interpolate


# model_weights_path = "/home/maoyj/projects/Only-Noisy-Training-main/whiteNe2Ne/Weights/dc10_model_25.pth"
model_weights_path = "./white_IDDP_DCUnet10_cTSTM/Weights/dc20_model_15.pth"
SAMPLE_RATE = 48000
# N_FFT = (SAMPLE_RATE * 64) // 1000 
# HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 
N_FFT = 1022
HOP_LENGTH = 256
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Test on GPU')

else:
    print('Test on CPU')


wonky_samples = []


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
                x_est = net(noisy.to(DEVICE), n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE)
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
        
        # fixed len
        # self.max_len = 165000
        self.max_len = 65280

    
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
        print("noisy", x_noisy.shape)
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True)
        print("stft", x_noisy_stft.shape)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True)
        
        return x_noisy_stft, x_clean_stft
        
    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        
        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)
        
        return output

DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')
dcunet10 = DCUnet10_cTSTM(N_FFT, HOP_LENGTH, DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(dcunet10.parameters())

checkpoint = torch.load(model_weights_path,
                        map_location=torch.device('cpu')
                       )


test_noisy_files = sorted(list(Path("Samples/Sample_Test_Input").rglob('*.wav')))
test_clean_files = sorted(list(Path("Samples/Sample_Test_Target").rglob('*.wav')))

test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)

dcunet10.load_state_dict(checkpoint)
# net = dcunet10.to(DEVICE)
testmet = getMetricsonLoader(test_loader_single_unshuffled,dcunet10)    # again, modified cuz im loading
with open("./results/results_IDDP.txt","w+") as f:
                f.write("iddp : \n")
                f.write(str(testmet))
                f.write("\n")
index = 4

dcunet10.eval()

test_loader_single_unshuffled_iter = iter(test_loader_single_unshuffled)

# x_n, x_c = next(test_loader_single_unshuffled_iter)
for i in range(index):
    x_n, x_c = next(test_loader_single_unshuffled_iter)
    x_est = dcunet10(x_n.to(DEVICE), N_FFT, HOP_LENGTH, DEVICE)

    x_est_np = x_est[0].view(-1).detach().cpu().numpy()
    x_c_np = torch.istft(torch.squeeze(x_c[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
    x_n_np = torch.istft(torch.squeeze(x_n[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()


    metrics = AudioMetrics(x_c_np, x_est_np, SAMPLE_RATE)
    print(metrics.display())

    # plt.savefig('./results/{}_x_n_.png'.format(i), x_n)
    # plt.savefig('./results/{}_x_est.png'.format(i), x_est)
    # plt.savefig('./results/{}_x_c.png'.format(i), x_c)
    fig1 = plt.figure(1)
    plt.plot(x_n_np)
    fig1.savefig('./results/{}_x_n_.png'.format(i))
    plt.close()
    fig2 = plt.figure(2)
    plt.plot(x_est_np)
    fig2.savefig('./results/{}_x_est.png'.format(i))
    plt.close()
    fig3 = plt.figure(3)
    plt.plot(x_c_np)
    fig3.savefig('./results/{}_x_c.png'.format(i))
    plt.close()


    noise_addition_utils.save_audio_file(np_array=x_est_np,file_path=Path("white_IDDP_DCUnet10_cTSTM/Samples/denoised.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
    noise_addition_utils.save_audio_file(np_array=x_c_np,file_path=Path("white_IDDP_DCUnet10_cTSTM/Samples/clean.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
    noise_addition_utils.save_audio_file(np_array=x_n_np,file_path=Path("white_IDDP_DCUnet10_cTSTM/Samples/noisy.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)