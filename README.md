# Multi-Information-Aware Speech Enhancement through Self-Supervised Learning
Source code for the paper titled "Multi-Information-Aware Speech Enhancement through Self-Supervised Learning". In this paper, we propose a self-supervised speech enhancement model, called Multi-Information-Aware Speech Enhancement (MIA-SE), to address these challenges. We introduce a novel self-supervised training strategy that performs denoising on one input data twice, utilizing the first output of the denoiser as an Implicit Deep Denoiser Prior (IDDP) to supervise the subsequent denoising process. Additionally, MIA-SE incorporates an encoder-decoder denoiser architecture based on a complex ratio masking strategy to extract phase and magnitude features simultaneously. To incorporate sequence context information for better embedding, we integrate transformer modules with multi-head attention mechanisms into the denoiser. The training process is guided by a newly formulated loss function to ensure successful and effective learning. Experimental results on synthetic and real-world noise databases demonstrate the effectiveness of MIA-SE, particularly in scenarios where paired training data is unavailable.

## Python Requirements
We recommend using Python 3.8.8. The package versions are in requirements.txt. We recommend using the Conda package manager to install dependencies.
```
conda create --name <env> --file requirements.txt
```

## Dataset 
We use 2 standard datasets; 'UrbanSound8K'(for real-world noise samples), and 'Voice Bank + DEMAND'(for speech samples). Please download the datasets from urbansounddataset.weebly.com/urbansound8k.html and datashare.ed.ac.uk/handle/10283/2791 respectively. 
To train a White noise denoising model, run the script:
```
python white_noise_dataset_generator.py
```

To train a UrbanSound noise class denoising model, run the script, and select the noise class:
```
python urban_sound_noise_dataset_generator.py

0 : air_conditioner
1 : car_horn
2 : children_playing
3 : dog_bark
4 : drilling
5 : engine_idling
6 : gun_shot
7 : jackhammer
8 : siren
9 : street_music
```
The train and test datasets for the specified noise will be generated in the 'Datasets' directory.

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

