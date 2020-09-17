# About Nvidia
The unofficial Nvidia products intro, to integrate the information scattered on the web, you may also want to access the official [one](https://developer.nvidia.com/) and the developer [forum](https://devtalk.nvidia.com).

### The ~~Distributed~~ `Nvidia` on Github
+ [Nvidia Research on Github](https://github.com/NVlabs) where you may find the work like [
Multimodal Unsupervised Image-to-Image Translation](https://github.com/NVlabs/MUNIT) for visual computing.

+ [NVIDIA Corporation on Github](https://github.com/Nvidia) where you can find products like [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and [NCCL](https://github.com/NVIDIA/nccl) to support your daily work, also [Sample code for CUDA 10](https://github.com/NVIDIA/cuda-samples).

+ [NVIDIA Jetson on Github](https://github.com/NVIDIA-Jetson) for roboticist and engineers, you may find resources about, Jetson, ROS and TensorRT to make your robots smarter and do things quicker and better, pls also see [Jetson-Inference](https://github.com/dusty-nv/jetson-inference) for more concrete examples.
+ [NVIDIA Deep Learning Accelerator (NVDLA) on Github](https://github.com/nvdla/) for free and open architecture that promotes a standard way to design deep learning inference accelerators.
+ [RAPIDS on Github](https://github.com/RAPIDSai) for GPU accelerated data-science, currently you may find the dataframe package on GPU both for C-[libgdf](https://github.com/rapidsai/libgdf) and for python-[pygdf](https://github.com/rapidsai/pygdf), also [cuml](https://github.com/rapidsai/cuml) library.
+ [NVIDIA GameWorks on Github](https://github.com/NVIDIAGameWorks) for game developers where you may find universal tools like [PhyX](https://github.com/NVIDIAGameWorks/PhysX) for gaming&engineering&research.

### On the Cloud
+ The GPU cards are accessible on cloud platform like AWS, Azure and Google Cloud for processing Compute-Intensive jobs, more specifically, there are products like [NVIDIA Volta Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B076K31M1S)  on AWS which is ready-to-work with just few minutes of setup.

+ As docker is widely used in different senarios of research and industry, the [Nvidia GPU Cloud](https://ngc.nvidia.com/) was prepared to provide ready-to-use docker images with various optimized software toolkits for applications like deep learning and high performance computing. To make your life easier combined with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/) and a customized [K8S](https://github.com/NVIDIA/kubernetes).

### Toolkits

All tools are available at [Nvidia Developer Homepage](https://developer.nvidia.com/), navigation bar would lead you to different application fields to specify which one suits your needs well, most of them are not open-sourced but freely available for developers.

+ [CUDA](https://developer.nvidia.com/cuda-zone) is the platform for general-purpose GPU programming, libraries like cuBLAS, cuRAND, cuSOLVER, AmgX, cuSPARSE, CUDA Math Library for linear algebra, cuFFT, NVIDIA performance Primitives and NVIDIA Codec SDK for signal/image/video processing, NCCL, nvGRAPH and Thrust for ready-to-use parallel algorithms were all integrated into the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). And many open-source tools(like OpenCV) were well supported using CUDA toolkits, you may enable the feature by building them from source or download the corresponding version like `tensorflow-gpu` versus `tensorflow`. CUDA is available for C/C++/Fortran programming, pyCUDA is separately maintained and you can check the content by clicking links on [https://developer.nvidia.com/pycuda](https://developer.nvidia.com/pycuda).
+ [OpenACC](https://developer.nvidia.com/openacc) is a good option if you would like to transfer partial of the code from the former project to GPU for acceleration, by adding directives rather than rewrite the whole project from scratch, you can also call the functions built in libraries like cuBLAS to replace current CPU version function call to further improve the performance.
