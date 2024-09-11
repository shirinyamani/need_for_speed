# Speculative Decoding 

# Why this works?
Most of the work getting done is **NOT** about compputation, but its actually about all those read/writes to memory access.
Bc whats happening is that the input lives on the memory and when you do any computation, it has to travel to the GPU/ to all the caches and registers to do the computation and then back to the memory. This is a very slow process. 
![alt text](image.png)

So each time we are doing round trips which is slow and very expensive. SO the idea is basically we gonna do a single trip to GPU and while that memory or at least a chunk of it is in the GPU, we are gonna do as much computation as possible and then we gonna load back the results to the memory.