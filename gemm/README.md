# Gemm

## 优化流程
1. 01_naive: baseline实现;
2. 02_mem_coalesce: 保证每个warp的global memory访问连续;
    - 参考资料: https://zhuanlan.zhihu.com/p/300785893
3. 

参考资料：
- https://siboehm.com/articles/22/CUDA-MMM