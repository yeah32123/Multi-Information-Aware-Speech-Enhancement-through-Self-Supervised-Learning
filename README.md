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

## Training
You can choose the following three models for training: DCUnet10,DCUnet10_rTSTM and DCUnet10_cTSTM. DCUnet10 represents our speech denoising approach utilizing a complex U-Net architecture without a Time Step Transform Module (TSTM); DCUnet10_rTSTM represents our novel strategy with a real-valued TSTM (rTSTM) between the complex U-Net; DCUnet10_cTSTM represents our novel strategy with a complex-valued TSTM (cTSTM) between the complex U-Net. You can train the model by running the following script:
```
python train.py
```

## Special thanks to the following repositories:
* https://github.com/pheepa/DCUnet
* https://github.com/ludlows/python-pesq
* https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data
* https://github.com/liqingchunnnn/Only-Noisy-Training

## References
[1] Abd  El-Fattah,   M.,  Dessouky,  M.I.,  Diab,  S.,  Abd  El-Samie,   F., 2008. Speech enhancement using an adaptive wiener filtering approach. Progress In Electromagnetics Research M 4, 167–184.

[2]  Boll, S., 1979. Suppression of acoustic noise in speech using spectral sub- traction. IEEE Transactions on acoustics, speech, and signal processing 27, 113–120.

[3]  Chen, X., He, K., 2021. Exploring simple siamese representation learn- ing, in:  Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15750–15758.

[4]  Choi, H.S., Kim, J.H., Huh, J., Kim, A., Ha, J.W., Lee, K., 2019. Phase- aware speech enhancement with deep complex u-net, in:  International Conference on Learning Representations.

[5]  Defossez, A., Synnaeve, G., Adi, Y., 2020.  Real time speech enhance- ment in the waveform domain. arXiv preprint arXiv:2006.12847 .

[6]  Erdogan, H., Hershey, J.R., Watanabe, S., Le Roux, J., 2015.  Phase- sensitive and recognition-boosted speech separation using deep recurrent neural networks, in: 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE. pp. 708–712.

[7]  Feng, H., Wang, L., Wang, Y., Huang, H., 2022. Learnability enhance- ment for low-light raw denoising:  Where paired real data meets noise modeling, in:  Proceedings  of the 30th ACM International Conference on Multimedia, pp. 1436–1444.

[8]  Fu, S.W., Liao, C.F., Tsao, Y., Lin, S.D., 2019.  Metricgan:  Genera- tive adversarial networks based black-box metric scores optimization for speech enhancement, in: International Conference on Machine Learning, PMLR. pp. 2031–2041.

[9] Fujimura, T., Koizumi, Y., Yatabe, K., Miyazaki, R., 2021. Noisy-target training:  A training strategy for dnn-based speech enhancement with- out clean speech, in: 2021 29th European Signal Processing Conference (EUSIPCO), IEEE. pp. 436–440.

[10]  Gandelsman, Y.,  Shocher, A., Irani, M., 2019.   ” double-dip”:  unsu- pervised image decomposition via coupled deep-image-priors, in:  Pro- ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11026–11035.

[11]  Gong, K., Catana, C., Qi, J., Li, Q., 2018.  Pet image reconstruction using deep image prior. IEEE transactions on medical imaging 38, 1655– 1665.

[12]  Guan, C., Chen, Y., Wu, B., 1993. Direct modulation on lpc coefficients with application to speech enhancement and improving the performance of speech recognition in noise, in:  1993 IEEE International Conference on Acoustics, Speech, and Signal Processing, IEEE. pp. 107–110.

[13]  Hu, G., Wang, D., 2004.  Monaural speech segregation based on pitch tracking and amplitude modulation. IEEE Transactions on neural net- works 15, 1135–1150.

[14]  Huang,  P.S.,  Kim,  M.,  Hasegawa-Johnson,  M.,  Smaragdis,  P.,  2015. Joint  optimization  of masks  and  deep  recurrent  neural  networks  for monaural  source  separation.     IEEE/ACM  Transactions  on  Audio, Speech, and Language Processing 23, 2136–2147.

[15]  Huang, T., Li, S., Jia, X., Lu, H., Liu, J., 2021.  Neighbor2neighbor: Self-supervised denoising from single noisy images, in:  Proceedings  of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14781–14790.

[16]  Kashyap, M.M., Tambwekar, A., Manohara, K., Natarajan, S., 2021. Speech Denoising Without Clean Training Data:  A Noise2Noise Ap- proach,  in:   Proc.  Interspeech  2021,  pp.  2716–2720.    doi:10.21437/ Interspeech.2021-1130.

[17] Kumar, A., Braud, T., Lee, L.H., Hui, P., 2021. Theophany: Multimodal speech augmentation in instantaneous privacy channels, in:  Proceedings of the 29th ACM International Conference on Multimedia, pp. 2056– 2064.

[18]  Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., Aila, T., 2018.   Noise2noise:  Learning image restoration without clean data. arXiv preprint arXiv:1803.04189 .

[19]  Lin,  H.,  Zhuang, Y., Huang, Y., Ding, X.,  2022.   Self-supervised  sar despeckling powered by implicit deep denoiser prior.  IEEE Geoscience and Remote Sensing Letters 19, 1–5.

[20]  Lin,  Z.,  Zeng,  B.,  Hu,  H.,  Huang,  Y.,  Xu,  L.,  Yao,  Z.,  2023.    Sase: Self-adaptive noise distribution network for speech enhancement with federated learning using heterogeneous data. Knowledge-Based Systems 266, 110396.

[21]  Loizou, P.C., 2013.   Speech enhancement:  theory  and  practice. CRC press.

[22]  Luo, Y., Chen, Z., Yoshioka, T., 2020.  Dual-path rnn:  efficient long sequence modeling for time-domain single-channel speech separation, in: ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE. pp. 46–50.

[23]  Mohammadiha,  N.,  Smaragdis,  P.,  Leijon,  A.,  2013.   Supervised  and unsupervised speech enhancement using nonnegative matrix factoriza- tion.  IEEE Transactions on Audio, Speech, and Language Processing 21, 2140–2151.

[24]  Narayanan, A., Wang, D., 2013. Ideal ratio mask estimation using deep neural networks for robust speech recognition, in:  2013 IEEE Interna- tional Conference on Acoustics, Speech and Signal Processing, IEEE. pp. 7092–7096.

[25]  Niresi,  K.F.,  Chi,  C.Y.,  2022.   Unsupervised  hyperspectral  denoising based on deep image prior and least favorable distribution. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 15, 5967–5983.

[26]  Ohri, K., Kumar, M., 2021. Review on self-supervised image recognition using deep neural networks. Knowledge-Based Systems 224, 107090.

[27]  O’Shaughnessy, D., 2008. Automatic speech recognition:  History, meth- ods and challenges. Pattern Recognition 41, 2965–2979.

[28]  Paliwal,  K.,  Basu,  A.,  1987.    A  speech enhancement  method based on kalman filtering, in: ICASSP’87. IEEE International Conference on Acoustics, Speech, and Signal Processing, IEEE. pp. 177–180.

[29]  Pandey, A., Wang, D., 2019. Tcnn: Temporal convolutional neural net- work for real-time speech enhancement in the time domain, in:  ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Sig- nal Processing (ICASSP), IEEE. pp. 6875–6879.

[30]  Pandey, A., Wang, D., 2020.  Densely connected neural network with dilated convolutions for real-time speech enhancement in the time do- main, in: ICASSP 2020-2020 IEEE International Conference on Acous- tics, Speech and Signal Processing (ICASSP), IEEE. pp. 6629–6633.

[31]  Park, S.R., Lee, J., 2016. A fully convolutional neural network for speech enhancement. arXiv preprint arXiv:1609.07132 .

[32]  Pascual, S., Bonafonte, A., Serra, J., 2017. Segan:  Speech enhancement generative adversarial network. arXiv preprint arXiv:1703.09452 .

[33]  Rethage, D., Pons, J., Serra, X., 2018. A wavenet for speech denoising, in: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE. pp. 5069–5073.

[34]  Salamon, J., Jacoby, C., Bello, J.P., 2014. A dataset and taxonomy for urban sound research, in:  Proceedings of the 22nd ACM international conference on Multimedia, pp. 1041–1044.

[35]  Saleem, N., Khattak, M.I., 2019.  A review of supervised learning algo- rithms for single channel speech enhancement. International Journal of Speech Technology 22, 1051–1075.

[36]  Serr`a,  J.,  Pascual,  S.,  Pons,  J.,  Araz,  R.O.,  Scaini,  D.,  2022.    Uni- versal speech enhancement with score-based diffusion.  arXiv preprint arXiv:2206.03065 .

[37]  Subakan,  C., Ravanelli, M., Cornell, S., Bronzi, M., Zhong, J., 2021. Attention is all you need in speech separation, in:  ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Pro- cessing (ICASSP), IEEE. pp. 21–25.

[38]  Sun, K., Zhang, X., 2021.  Ultrase:  single-channel speech enhancement using ultrasound, in: Proceedings of the 27th annual international con- ference on mobile computing and networking, pp. 160–173.

[39]  Sun, Z., Li, Y., Jiang, H., Chen, F., Xie, X., Wang, Z., 2020.  A su- pervised  speech  enhancement  method  for  smartphone-based  binaural hearing aids.  IEEE Transactions on Biomedical Circuits and Systems 14, 951–960.

[40] Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J.F., Mehri, S., Rostamzadeh, N., Bengio, Y., Pal, C.J., 2017. Deep complex networks. arXiv preprint arXiv:1705.09792 .

[41]  Ulyanov, D., Vedaldi, A., Lempitsky, V., 2018.  Deep image prior, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 9446–9454.

[42] Valentini-Botinhao,  C.,  Wang,  X.,  Takaki,  S.,  Yamagishi,  J.,  2016. Speech enhancement for a noise-robust text-to-speech synthesis system using deep recurrent neural networks., in: Interspeech, pp. 352–356.

[43] Wang, D., 2008. Time-frequency masking for speech separation and its potential for hearing aid design. Trends in amplification 12, 332–353.

[44] Wang, D., Brown, G.J., 2006.  Computational auditory scene analysis: Principles, algorithms, and applications. Wiley-IEEE press.

[45] Wang, H., Liu, Z., Ge, Y., Peng, D., 2022. Self-supervised signal repre- sentation learning for machinery fault diagnosis under limited annota- tion data. Knowledge-Based Systems 239, 107978.

[46] Wang,  K.,  He,  B.,  Zhu,  W.P.,  2021a.   Caunet:   Context-aware  u-net for speech enhancement in time domain, in:  2021  IEEE International Symposium on Circuits and Systems (ISCAS), IEEE. pp. 1–5.

[47] Wang, K., He, B., Zhu, W.P., 2021b.  Tstnn:  Two-stage transformer based neural network for speech enhancement in the time domain, in: ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE. pp. 7098–7102.

[48] Wang, Y., Narayanan, A., Wang, D., 2014.  On training targets for su- pervised speech separation. IEEE/ACM transactions on audio, speech, and language processing 22, 1849–1858.

[49] Williamson, D.S., Wang, Y., Wang, D., 2015. Complex ratio masking for monaural speech separation. IEEE/ACM transactions on audio, speech, and language processing 24, 483–492.

[50] Wisdom, S., Tzinis, E., Erdogan, H., Weiss, R., Wilson, K., Hershey, J., 2020.  Unsupervised sound separation using mixture invariant training. Advances in Neural Information Processing Systems 33, 3846–3857.

[51] Yang, X., Zhang, Z., Cui, R., 2022.   Timeclr:  A self-supervised con- trastive learning framework for univariate time series representation. Knowledge-Based Systems 245, 108606.

[52] Yu, G., Li, A., Wang, H., Wang, Y., Ke, Y., Zheng, C., 2022. Dbt-net: Dual-branch federative magnitude and phase estimation with attention- in-attention transformer for monaural speech enhancement. IEEE/ACM Transactions on Audio, Speech, and Language Processing 30, 2629–2644.

[53]  Zhu, H., Niu, Y., Fu, D., Wang, H., 2021. Musicbert: A self-supervised learning of music representation, in:  Proceedings of the 29th ACM In- ternational Conference on Multimedia, pp. 3955–3963.

