# Gemm

## GPU信息
```
Device ID: 0
Name: NVIDIA A800-SXM4-80GB
Compute Capability: 8.0
memoryBusWidth: 5120
maxThreadsPerBlock: 1024
maxThreadsPerMultiProcessor: 2048
maxRegsPerBlock: 65536
maxRegsPerMultiProcessor: 65536
totalGlobalMem: 81251MB
sharedMemPerBlock: 48KB
sharedMemPerMultiprocessor: 164KB
totalConstMem: 64KB
multiProcessorCount: 108
Warp Size: 32
```

## 优化流程
1. 01_naive: baseline实现;
2. 02_mem_coalesce: 保证每个warp的global memory访问连续;
    - 参考资料: https://zhuanlan.zhihu.com/p/300785893
3. 

参考资料：
- https://siboehm.com/articles/22/CUDA-MMM