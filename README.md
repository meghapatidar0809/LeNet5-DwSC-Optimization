# LeNet-5 Implementation, Optimization and Parallelism

### Overview  
This project implements the **LeNet-5** architecture, its optimization, and various parallel training strategies using **PyTorch** and trains it on the **MNIST** dataset. It builds upon a standard **LeNet-5** implementation and a version optimized with **depthwise separable convolutions**. The project also investigates data and model parallelism to enhance training efficiency and to analyze their impact on model performance.  

### Features  
- **LeNet-5 Implementation**: Standard architecture trained on MNIST dataset.  
- **Depthwise Separable Convolutions**: An optimized version of LeNet-5 where standard convolutional layers are replaced to analyze their impact on model performance.
- **Data Parallelism**:Implemented using **all-reduce** and **ring-all-reduce** to distribute the training load across multiple CPU cores.
- **Model Parallelism**:he LeNet-5 network is partitioned, and its layers are distributed across multiple cores.
- **Performance Analysis**: Plots accuracy vs. epochs and loss vs. epochs for all implemented models and strategies, allowing for a detailed comparison. 

### Tools & Libraries  
- **Python, PyTorch, NumPy, Matplotlib**
- **torch.distributed**: Used for implementing data and model parallelism.

### Dataset  
- **MNIST**: A dataset of handwritten digits (0-9) with **60,000 training** and **10,000 testing** images.


### Reference
- **Paper**: https://ieeexplore.ieee.org/document/726791

