# Speculative Decoding 

# Why this works?
Most of the work getting done is **NOT** about compputation, but its actually about all those read/writes to memory access.
Bc whats happening is that the input lives on the memory and when you do any computation, it has to travel to the GPU/ to all the caches and registers to do the computation and then back to the memory. This is a very slow process. 
![alt text](img/image.png)

So each time we are doing round trips which is slow and very expensive. SO the idea is basically we gonna do a single trip to GPU and while that memory or at least a chunk of it is in the GPU, we are gonna do as much computation as possible and then we gonna load back the results to the memory.

Now the clever idea is to use a small and cheap draft model to first generate a candidate sequence of K tokens - a "draft". Then we feed all of these together through the big model in a batch. This is almost as fast as feeding in just one token, per the above. Then we go from left to right over the logits predicted by the model and sample tokens. Any sample that agrees with the draft allows us to immediately skip forward to the next token. If there is a disagreement then we throw the draft away and eat the cost of doing some throwaway work (sampling the draft and the forward passing for all the later tokens).

The reason this works in practice is that most of the time the draft tokens get accepted, because they are easy, so even a much smaller draft model gets them. As these easy tokens get accepted, we skip through those parts in leaps. The hard tokens where the big model disagrees "fall back" to original speed, but actually a bit slower because of all the extra work.


# Why this works mathematically?

Speculative decoding's mathematical foundation is rooted in rejection sampling, a Monte Carlo method used to generate samples from a target distribution when direct sampling is difficult.

# Why so magically we can re-construct the probability distribution of the large model from the smaller model?

In the following I try to explain why sampling from another smaller model which is somewhat close to the large model can actually reconstruct the probability distribution of the large model.

Imagine we have a crazy complicated funtion that we cannot access it directly to be able to sample from it. One approach in such situation is to instead sample from a simpler model that is somewhat close to the complicated one. 

If we decide to go for such approach, then the chosen proposed model has to have two important properties:

1. The proposed model should be somewhat close to the large model, i.e. the proposed model must be able to capture the overall shape of the distribution of the large model. In other words, it must be able to generate the most common and likely tokens that the large model would generate. This ensures that the proposed model is a good approximation of the large model's behavior for the majority of the tokens.

2. The proposed model must be easy to sample from, i.e. we must be able to generate tokens from the proposed model efficiently.

