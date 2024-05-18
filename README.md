# Multi-Information-Aware Speech Enhancement through Self-Supervised Learning
Source code for the paper titled "Multi-Information-Aware Speech Enhancement through Self-Supervised Learning". In this paper, we propose a self-supervised speech enhancement model, called Multi-Information-Aware Speech Enhancement (MIA-SE), to address these challenges. We introduce a novel self-supervised training strategy that performs denoising on one input data twice, utilizing the first output of the denoiser as an Implicit Deep Denoiser Prior (IDDP) to supervise the subsequent denoising process. Additionally, MIA-SE incorporates an encoder-decoder denoiser architecture based on a complex ratio masking strategy to extract phase and magnitude features simultaneously. To incorporate sequence context information for better embedding, we integrate transformer modules with multi-head attention mechanisms into the denoiser. The training process is guided by a newly formulated loss function to ensure successful and effective learning. Experimental results on synthetic and real-world noise databases demonstrate the effectiveness of MIA-SE, particularly in scenarios where paired training data is unavailable.

## Python Requirements
待添加

## Dataset 
待添加

## Training a New Model
待添加

## Testing Model Inference on Pretrained Weights
待添加


## Special thanks to the following repositories:
* https://github.com/pheepa/DCUnet
* https://github.com/ludlows/python-pesq
* https://github.com/mpariente/pystoi
待添加
## References
待添加

