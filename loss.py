import torch
import torch.nn as nn
from loss_utils import mse_loss, stftm_loss, reg_loss

time_loss = mse_loss()
freq_loss = stftm_loss()

def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False)
    for j, seq_len in enumerate(nframes):
        print("j", j)
        print("seq_len", seq_len)
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

def wsdr_tf(x, y_pred, y_true):
    if(y_true.shape[0] == 2):
            nframes = [y_true.shape[1],y_true.shape[1]]   # nframes: [32512, 32512]
    else:
            nframes = [y_true.shape[1]]

    loss_mask = compLossMask(y_true.unsqueeze(1), nframes)   
    loss_mask = loss_mask.float().cuda() 
    loss_time = time_loss(y_pred.unsqueeze(1), y_true.unsqueeze(1), loss_mask)
    loss_freq = freq_loss(y_pred.unsqueeze(1), y_true.unsqueeze(1), loss_mask)
    loss1 = (0.8 * loss_time + 0.2 * loss_freq)/600
    loss = loss1 + wsdr_fn(x, y_pred, y_true)

    return loss