## Nano Model

How small is really possible?

### Best Model: NanoGRU

Before reparameterization: 19975 parameters using 10.6 MB of memory for each spectrogram.

After reparameterization: 9471 parameters using 3.1 MB of memory for each spectrogram.

#### Confusion Matrix

```
 tensor([[ 970,   40,   35,    0,    3,    0,    0,    0,    0],
        [ 126,  250,  193,    0,    2,    0,    0,    5,    0],
        [  38,  108, 1237,    0,    1,    0,    3,    1,    0],
        [   8,    4,    0,    0,    0,    0,    0,    0,    0],
        [   1,    3,    0,    0,   40,    0,    0,    0,    0],
        [   0,    0,    0,    0,    0,   70,    2,    0,    0],
        [   0,    2,    5,    0,    0,    4,   73,    0,    0],
        [   0,    0,    1,    0,    0,    0,    0,   39,    0],
        [   0,    0,   32,    0,    0,    0,    0,    0,    0]],
       device='cuda:1')
```

#### Per-Class Accuracy

```
  class 0 (None                  ):  92.557%
  class 1 (None                  ):  43.403%
  class 2 (None                  ):  89.121%
  class 3 (None                  ):   0.000%
  class 4 (Chirp, high distance  ):  90.909%
  class 5 (Chirp, medium distance):  97.222%
  class 6 (Chirp, small distance ):  86.905%
  class 7 (Cigarette lighter 1   ):  97.500%
  class 8 (Cigarette lighter 2   ):   0.000%
```

#### Metrics

```
test loss: 0.501
test f1: 0.657
test acc:  81.280%
```

#### Throughput

Latency is per item (single spectrogram) of shape (1, 1, 512, 243) after reparameterization.

| Accelerator         | Dtype    | Latency (ms) |
|---------------------|----------|--------------|
| RTX 4090            | bfloat16 | 0.023        |
| RTX 4090            | fp32     | 0.024        |
| RTX 2080 Ti         | fp32     | 0.072        |
| Threadripper 5975WX | fp32     | 0.606        |
| Threadripper 2990WX | fp32     | 1.069        |

### Experiment Log

| Model                    | par (M) | memory (MB) | madds (G) | TLoss | F1    | Acc% | Changes |
|--------------------------|---------|-------------|-----------|-------|-------|------|---------|
| GRU                      |    0.8  |             |           | 0.619 | 0.575 | 77.0 |  |
| Conv(k3) + GRU(128)      |    4.7  |             |           | 0.533 | 0.636 | 78.0 |  |
| Conv(k3) + GRU(256)      |    4.7  |             |           |       |       | 79.5 |  |
| Conv(k11) + GRU(128)     |    2.0  |             |           | 0.508 | 0.647 | 80.3 |  |
| Conv(k11) + GRU(128)     |    2.0  |             |           | 0.507 | 0.648 | 80.2 | num_conv_branches: 4->1, good |
| Conv(k11) + GRU(128)     |    5.13 |        90.1 |     0.517 | 0.521 | 0.639 | 78.6 | planes: 16->48, overtrained   |
| Conv(k7) + GRU(128)      |    1.97 |        31.5 |     0.148 | 0.544 | 0.619 | 78.2 | stem: k11->k7, worse   |
| Stem(11) + minGRU(128)   |    0.53 |  25.7 (8.7) |     0.065 | 0.543 | 0.635 | 78.1 | gru(128,n=2) -> minGRU(128,n=1), worse |
| Stem(11) + minGRU(128)   |    0.56 |  26.0 (9.0) |     0.065 | 0.530 | 0.617 | 78.9 | gru(128) -> minGRU(128), slightly worse |
| Stem(11) + minLSTM(128)  |    0.84 | 27.2 (10.2) |     0.065 | 0.558 | 0.618 | 77.9 | minGRU -> minLSTM, no imp |
| Stem(11) + minGRU(256)   |    1.19 | 28.7 (11.7) |     0.066 | 0.527 | 0.651 | 79.5 | minGRU(128) -> minGRU(256), good |
| Stem(11) + GRU(256,b=F)  |    2.17 | 32.3 (15.3) |     0.196 | 0.505 | 0.659 | 80.5 | GRU(128,bi=True) vs GRU(256), new baseline |
| S + Conv(13,s=2) + GRU   |    2.18 | 35.8 (16.3) |     0.144 | 0.517 | 0.657 | 79.2 | Add ReParamLargeKernelConv, losing too much time res? |
| S + Conv(13,s=1) + GRU   |    3.75 | 52.6 (25.6) |     0.343 | 0.515 | 0.646 | 79.8 | no stride in Large  |
| S* + Conv(13,s=1) + GRU  |    3.75 | 52.6 (25.6) |     0.343 | 0.505 | 0.678 | 80.8 | Replace expand w/LargeExpand, good |
| FastViT(16planes, mlp=4) |    1.55 | 92.8 (49.8) |     0.261 | 0.573 | 0.615 | 77.0 | Disappointing ViT, LR too high? |
| FastViT(48planes, mlp=3) |    4.00 |   245 (130) |     1.350 | 0.583 | 0.606 | 76.5 | scaled up to t8 params, lr=5e-5 |
| S*Conv(13)PoolGRU        |    0.63 | 40.1 (13.1) |     0.153 | 0.555 | 0.611 | 77.6 | Pool freq before GRU, way smaller input size |
| S*Conv11,2x(conv,pool)GRU|    0.66 | 54.3 (14.1) |     0.189 | 0.535 | 0.626 | 78.3 | Use avgpool to stride down only in freq before GRU |
| S*Conv11,2x(conv,pool)GRU|    0.66 | 54.3 (14.1) |     0.189 | 0.507 | 0.640 | 79.8 | More noise in train_dataloader. Double train time. |
| FastViT(16),Atten,Tweaks |    1.03 | 205.0(97.5) |     0.448 | 0.530 | 0.681 | 80.3 | Freq pooling in stem, bs=48. LR scheduler 300epochs. |
| FastViT(16),Atten,Tweaks |    1.03 | 205.0(97.5) |     0.448 | 0.494 | 0.681 | 80.4 | CyclicLR, randomcutout! |
| FastViT(16),Atten,Tweaks |    1.03 | 205.0(97.5) |     0.448 | 0.524 | 0.644 | 78.6 | 100% cut, 100% noised |
| FastViT(16),Atten,Tweaks |    1.03 | 205.0(97.5) |     0.448 | 0.530 | 0.621 | 78.5 | 100% cut, lower noise |
| S*Conv11,2x(conv,pool)GRU|    0.66 | 54.3 (14.1) |     0.189 | 0.493 | 0.657 | 80.6 | "", Good Baseline! |
| ResNet18                 |    11.7 |       150.0 |     4.634 | 0.513 | 0.684 | 79.9 | Re-run original baseline. |
| TinyGRU                  |    0.01 |  18.8 (4.1) |     0.051 | 0.546 | 0.615 | 77.6 | What is possible w/less than 10k params? Turns out a lot. |
| TinyGRU, AttentionPool   |    0.01 |  18.8 (4.1) |           | 0.549 | 0.587 | 77.3 | Failed EXP. |
| TinyGRU, Mods            |    0.01 | 65.2 (11.4) |     0.043 | 0.552 | 0.602 | 77.0 | 2x depth, 6planes, worse result |
| TinyGRU(13,16,34)        |    0.01 | 75.6 (10.6) |     0.097 | 0.539 | 0.607 | 78.3 | Decent for 9739 params. |
| TinyGRU(13,16,34)        |    0.01 | 75.6 (10.6) |     0.097 | 0.537 | 0.611 | 78.6 | DropPath instead of Dropout, Ran 600 epochs, bad idea. |
| TinyGRU(13,16,34)        |    0.01 | 75.6 (10.6) |     0.097 | 0.596 | 0.494 | 77.2 | LayerNorm before GRU. -> Worse. |
| TinyGRU(13,16,34)        |    0.01 | 75.6 (10.6) |     0.097 | 0.958 | 0.314 | 65.1 | LayerNorm across freqs before stem. -> Much much worse |
| TinyGRU(13,16,34)        |   0.009 | 69.6 (10.1) |     0.094 | 0.555 | 0.597 | 78.3 | Channel expansion in pointwise. **minGRU baseline!** |
| TinyGRU(13,16,38)        |    0.01 | 69.6 (10.1) |     0.094 | 0.227 | 0.575 | 77.5 | FocalLoss -> SLIGHTLY worse |
| TinyGRU(13,16,19)        |    0.01 | 69.5 (10.0) |     0.095 | 0.535 | 0.624 | 79.6 | GRU(19) instead of minGRU(39), simplify conv -> **GRU Baseline!** |
| TinyGRU(13,16,39)        |    0.01 | 69.6 (10.1) |     0.094 | 0.543 | 0.605 | 78.7 | minGRU(39) instead of GRU(19), simplify conv |
| MobileNetV3_Large        |    5.49 |       203.7 |     0.552 | 0.546 | 0.636 | 77.7 | Re-run original baseline. |
| TinyGRU(13,16,19)        |    0.01 | 69.5 (10.0) |     0.095 | 0.670 | 0.477 | 74.8 | 2xRandomErasing, RandomSharpness -> Bad |
| TinyGRU(13,16,19)        |    0.01 | 69.5 (10.0) |     0.095 | 0.540 | 0.608 | 78.5 | 2xRandomErasing -> about the same |
| MicroGRUv1(13, 20, 20)   |    0.01 | 90.2 (31.4) |     0.022 | 0.534 | 0.627 | 79.6 | Wow. More memory but even fewer params. |
| MicroGRUv2(3, 64, 64)    |    0.17 |       100.0 |     0.086 | 0.585 | 0.582 | 76.76 | same FLOPS as TinyGRU but 10x memory -> not great |
| TinyGRU(13,16,19)_basel  |    0.01 | 69.5 (10.0) |     0.095 | 0.537 | 0.620 | 78.8 | Add Equal-Weighted Category Loss -> no imp. |
| MicroGRUv3(15,32,32,mean)|    0.01 |  21.0 (9.9) |     0.007 | 0.535 | 0.625 | 79.1 | 100 epochs max (freq mean-vs-max) -> worse |
| MicroGRUv3(15,32,32,max) |    0.01 |  21.0 (9.9) |     0.007 | 0.532 | 0.633 | 79.6 | 100 epochs max (freq mean-vs-max) -> better |
| MicroGRUv3(15,32,64)     |    0.05 | 36.7 (11.0) |     0.011 | 0.523 | 0.617 | 79.5 | 100 epochs max (increase hidden 32->64) -> slightly better |
| MicroGRUv3(31,32,32)     |    0.02 | 27.4 (10.0) |     0.013 | 0.532 | 0.622 | 79.3 | 100 epochs max (larger k) -> same |
| MicroGRUv3(15,32,32)[223]|    0.04 | 37.4 (11.0) |     0.011 | 0.533 | 0.627 | 79.1 | 100 epochs max (more repeats) |
| MicroGRUv3(15,32,32)Large|    0.02 |  26.7 (4.6) |     0.061 | 0.532 | 0.630 | 79.7 | 100 epochs max (DYMicro -> LargeReparam stem) -> good |
| MicroGRUv3(15,32,32)Large|    0.02 |  26.7 (4.6) |     0.061 | 0.524 | 0.630 | 79.4 | -1 epochs -> **Micro Baseline!**  |
| MicroGRUv4(15,32,32)pe[123]|  0.02 |  21.4 (4.4) |     0.066 | 0.522 | 0.630 | 79.5 | patchembed -> vgood |
| MicroGRUv4(15,32,32)pe[234]|  0.03 |  38.5 (6.7) |     0.070 | 0.507 | 0.669 | 79.9 | patchembed -> vvgood -> **Micro Baseline** |
| MicroGRUv4(15,32,32)pe[234]|  0.03 |  38.5 (6.7) |     0.070 | 0.505 | 0.663 | 80.7 | bs (32->256) **Micro Baseline** |
| NanoGRU(15,16,18)pretimm |    0.02 |  19.6 (3.0) |     0.032 | 0.529 | 0.636 | 80.4 | **Nano Baseline** (pre timm) |
| NanoGRU(15,16,18)timm    |    0.01 |  10.6 (3.1) |     0.032 | 0.538 | 0.644 | 79.8 | timm convert |
| NanoGRU(15,16,18)timm    |    0.01 |  10.6 (3.1) |     0.032 | 0.529 | 0.644 | 80.2 | lkc_use_act=False, **Nano Baseline** |
| NanoGRU+mods             |    0.01 | 43.1 (14.9) |     0.135 | 0.551 | 0.650 | 79.8 | more channels, more depth, less strides -> bad |
| NanoGRU, initial(31, 15) |    0.01 |  10.6 (3.1) |     0.124 | 0.530 | 0.642 | 80.4 | Basically no change from baseline |
| NanoGRU, initial(11, 5)  |    0.01 |  10.6 (3.1) |     0.019 | 0.543 | 0.650 | 79.6 | Converged suprisingly slowly |
| NanoGRU+mods+minGRU      |    0.02 |  10.6 (3.1) |     0.032 | 0.504 | 0.645 | 80.4 | dont Reduce freq, minGRU; converged VERY fast. |
| NanoGRU+mods+GRU         |    0.02 |  10.6 (3.7) |     0.032 | 0.553 | 0.623 | 79.6 | dont Reduce freq, GRU; significantly worse than minGRU |
| NanoGRU(19)+Oversampling |    0.01 |  10.6 (3.6) |     0.032 | 0.565 | 0.650 | 77.4 | Oversample inference classes -> Increased low-snr detection but negatively impacted high-snr detection. |
| NanoGRU(18), Multihead   |    0.01 |  10.6 (3.6) |     0.041 | 0.181 | 0.617 | 77.4 | Multihead to predict (power, class) -> lower overall ACC |
| NanoGRU18+RandomTimeCrop |    0.01 |  10.6 (3.6) |     0.041 | 0.501 | 0.657 | 81.3 | **Nano Baseline** |
+droppath
| NanoGRU+DropPath         |    0.01 |  10.6 (3.6) |     0.041 | 0.518| 0.637 | 80.0 | Added DropPath(0.1) instead of Dropout(0.3) |

###  Insights

* Attempting to scale the log values (-135,0) into (0, 1) was a bad idea. Most of the input range is (0.6,0.9) and the result is very noisy val_loss.
* Adding more noise (-90, -40) results in less overfitting and higher accuracy.
* Randomcutout does a great job augmenting and preventing overfitting to our limited dataset.
* For the same parameter count, GRU is better than minGRU
* For the same hidden size, minGRU is better than (GRU/LSTM/minLSTM)
* Taking mean along time dimension after GRU better than attention pooling
* Stochastic Depth aka DropPath(0.1) is usually better than Dropout(0.3), but doubles training epochs
* Micro-Factorized large kernels make a lot of sense for PSDs since the kernels for time and freq are kept separate.
* Patch embedding works very well for downsampling between layers, a lesson from FastViT
* RandomOverSampler on classes (3,4,5,6,7,8) improved low-SNR performance but hurt overall accuracy b/c 90% of samples are in class (0, 1, 2).
    * Class 0 (92.2% -> 91.2%)
    * Class 1 (29.7% -> 26.4%)
    * Class 2 (89.0% -> 88.1%)
    * Class 3 ( 0.0% -> 25.0%) **classified 3/12 correctly, only 43 actual samples in training set**
    * Class 4 (95.4% -> 88.6%)
    * Class 5 (90.3% -> 84.7%)
    * Class 6 (83.3% -> 82.1%)
    * Class 7 (82.5% -> 85.0%)
    * Class 8 (21.9% -> 43.8%)
* Creating a multi-head model to separately predict power and class resulted in lower overall class accuracy. Predicted (Background, Interference, Chirp, Cig1, Cig2) as logits and (Low, Medium, High) power levels as (-1, 0, 1) separately. I converted back to single-class during testing for final metrics.
* Randomly cropping in time yielded big gains.