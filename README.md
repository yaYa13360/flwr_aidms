# flwr_aidms
- Implement flwr federated learning on aidms
- source: https://flower.ai/docs/framework/ref-api/flwr.html

## devices 
- server
    - cpu

- client 1
    - gpu-1 (NVIDIARTXA5000)

- client 2
    - gpu-0 (NVIDIARTXA5000)

- original 
    - without federated

## experiments
- classification
    - dataset: cifar10
    - model: EfficientNetB0
    - demo date: 231218

- segmentation
    - dataset: The Oxford-IIIT Pet Dataset
    - model: U-Net(MobileNetV2、pix2pix)
    - demo date: 240415