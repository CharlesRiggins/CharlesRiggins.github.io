---
title: "Explaining the Code of the vLLM Inference Engine"
date: 2024-04-09
draft: false
math: false
---

#### A casual look into the vLLM codebase

### vLLM

If you are familiar with large language models (LLMs), you probably have heard of the vLLM. I believe the “v” in its name stands for *virtual* because it borrows the concept of virtual memory from the operating systems.

The vLLM stands out for its remarkable speed, offering an order of magnitude faster throughput compared to traditional inference libraries like the *transformers*.

The vLLM is fast because it processes the data in large batches. Simple but not simplistic. To achieve this, it needs to have on-demand KV cache allocation and dynamic batching.

We’ll explore these features by examining the vLLM’s source code. Keep in mind that the vLLM is constantly evolving, with frequent updates to its codebase. Although we’re discussing version v0.4.0 here, it may be superseded by the time you read this article. Nonetheless, the core functionality we’ll describe should still provide a comprehensive understanding of how this powerful inference engine operates.

### Online and Offline inference

The vLLM operates in two modes: online and offline. In offline inference, it functions similarly to a PyTorch module, allowing you to run it with input data. Online inference, on the other hand, functions more like a server. Once started, it awaits requests from clients and can handle multiple requests concurrently.

Despite apparent differences, both modes share the same underlying inference engine. For the sake of practicality, we will focus solely on online inference, as it is the mode commonly used in real-world production scenarios.

We will walk through five sections of the code: the server, engine initialization, new requests handling, the main loop of the engine, and the scheduler. We will not delve into the intricacies of model execution and kernel implementation here, but reserve them for future articles.

### The vLLM Server

The vLLM utilizes FastAPI to host its server. Within the server, an AsyncLLMEngine is instantiated. Despite its name, AsyncLLMEngine functions as a wrapper for the actual engine, named \_AsyncLLMEngine.

When the server receives a request to generate an output, it triggers the following function:

```
@app.post("/generate")  
async def generate(request: Request) -> Response:  
    """Generate completion for the request."""  
    request_dict = await request.json()  
    prompt = request_dict.pop("prompt")  
    stream = request_dict.pop("stream", False)  
    sampling_params = SamplingParams(**request_dict)  
    request_id = random_uuid()  
​  
    results_generator = engine.generate(prompt, sampling_params, request_id)  
​  
    # Streaming case  
    async def stream_results() -> AsyncGenerator[bytes, None]:  
        async for request_output in results_generator:  
            prompt = request_output.prompt  
            text_outputs = [  
                prompt + output.text for output in request_output.outputs  
            ]  
            ret = {"text": text_outputs}  
            yield (json.dumps(ret) + "\0").encode("utf-8")  
​  
    if stream:  
        return StreamingResponse(stream_results())  
​  
    # Non-streaming case  
    final_output = None  
    async for request_output in results_generator:  
        if await request.is_disconnected():  
            # Abort the request if the client disconnects.  
            await engine.abort(request_id)  
            return Response(status_code=499)  
        final_output = request_output  
​  
    assert final_output is not None  
    prompt = final_output.prompt  
    text_outputs = [prompt + output.text for output in final_output.outputs]  
    ret = {"text": text_outputs}  
    return JSONResponse(ret)
```

The engine generates one token in every step of inference. For streaming requests, the server returns generated tokens as soon as they are available, without waiting for the entire completion. Conversely, for non-streaming requests, the server waits until the entire text completion is generated before responding to the client.

### Handling of New Requests

As illustrated above, the server calls the *generate* method of the engine wrapper and gets a generator for output tokens.

```
results_generator = engine.generate(prompt, sampling_params, request_id)
```

Let’s see what the engine wrapper does in the *generate* method.

```
async def generate(  
    self,  
    prompt: Optional[str],  
    sampling_params: SamplingParams,  
    request_id: str,  
    prompt_token_ids: Optional[List[int]] = None,  
    lora_request: Optional[LoRARequest] = None,  
    multi_modal_data: Optional[MultiModalData] = None  
) -> AsyncIterator[RequestOutput]:  
​  
    arrival_time = time.time()  
​  
    try:  
        stream = await self.add_request(  
            request_id,  
            prompt,  
            sampling_params,  
            prompt_token_ids=prompt_token_ids,  
            arrival_time=arrival_time,  
            lora_request=lora_request,  
            multi_modal_data=multi_modal_data,  
        )  
​  
        async for request_output in stream:  
            yield request_output
```

The generator method simply yields from an iterator which it gets from calling the *add\_request* method.

Now look at the *add\_request* method.

```
async def add_request(  
        self,  
        request_id: str,  
        prompt: Optional[str],  
        sampling_params: SamplingParams,  
        prompt_token_ids: Optional[List[int]] = None,  
        arrival_time: Optional[float] = None,  
        lora_request: Optional[LoRARequest] = None,  
        multi_modal_data: Optional[MultiModalData] = None,  
    ) -> AsyncStream:  
      
        # Some trivial code omitted.  
          
        stream = self._request_tracker.add_request(  
            request_id,  
            prompt=prompt,  
            sampling_params=sampling_params,  
            prompt_token_ids=prompt_token_ids,  
            arrival_time=arrival_time,  
            lora_request=lora_request,  
            multi_modal_data=multi_modal_data,  
        )  
​  
        return stream
```

It passes the request to a request tracker and gets back a stream. A stream, in this context, is just another name for an asynchronous iterator.

```
def add_request(self, request_id: str,  
                **engine_add_request_kwargs) -> AsyncStream:  
    """Add a request to be sent to the engine on the next background  
    loop iteration."""  
    if request_id in self._request_streams:  
        raise KeyError(f"Request {request_id} already exists.")  
​  
    stream = AsyncStream(request_id)  
    self._new_requests.put_nowait((stream, {  
        "request_id": request_id,  
        **engine_add_request_kwargs  
    }))  
​  
    self.new_requests_event.set()  
​  
    return stream
```

The request tracker creates a stream for the request and pushes it into a queue. Then it sets up an event to inform the engine about the arrival of a new request. Finally, it returns the stream.

Note that the process of handling new requests is entirely separate from the engine’s operations. Its sole purpose is to enqueue requests. The engine runs concurrently in another thread (or coroutine), fetching requests from the queue.

### Engine Initialization

The initialization of the engine comprises the initializations of the workers, the cache engine, and the scheduler.

First, creating the workers.

The vLLM assigns a worker to manage each GPU, ensuring efficient parallelism. For instance, if you use 4 GPUs, it will create 4 corresponding workers. The worker handling the 0th GPU in the main process is referred to as the driver worker. Other workers are allocated to separate processes spawned from the main process.

Workers are responsible for GPU-related tasks. During engine initialization, they load the model weights to GPUs.

The second step is to set up the cache engine.

Each worker has its own cache engine. Each cache engine manages the memory allocated for KV cache storage in its GPU. To maximize the batch size, the cache engine seeks to occupy as much GPU memory as possible. The memory available for it equals the total GPU memory minus the size of model weights, the intermediate activation size, and a buffer (typically 10% of the total memory). The model size is already known but not the intermediate activation size, i.e., the largest memory ever taken by intermediate activations during inference. The vLLM determines this number by running dummy data and then profiling memory consumption. The size of the dummy data is determined by a parameter in the configuration, which by default is set to the largest context length the model supports.

```
def _init_cache(self) -> None:  
    # Get the maximum number of blocks that can be allocated on GPU and CPU.  
    num_blocks = self._run_workers(  
        "profile_num_available_blocks",  
        block_size=self.cache_config.block_size,  
        gpu_memory_utilization=self.cache_config.gpu_memory_utilization,  
        cpu_swap_space=self.cache_config.swap_space_bytes,  
        cache_dtype=self.cache_config.cache_dtype,  
    )  
​  
    num_gpu_blocks = min(b[0] for b in num_blocks)  
    num_cpu_blocks = min(b[1] for b in num_blocks)  
​  
    self.cache_config.num_gpu_blocks = num_gpu_blocks  
    self.cache_config.num_cpu_blocks = num_cpu_blocks  
​  
    # Initialize the cache.  
    self._run_workers("init_cache_engine", cache_config=self.cache_config)  
    # Warm up the model. This includes capturing the model into CUDA graph  
    # if enforce_eager is False.  
    self._run_workers("warm_up_model")
```

The vLLM calls *torch.cuda.mem\_get\_info()* to get the peak memory usage following the execution of dummy data. This usage should encompass both the size of the weights and the intermediate activations. The remaining GPU memory, subtracted by a memory buffer, yields the memory available for the cache engine.

The vLLM manages GPU memory in blocks, just like the operating systems. If additional memory is required to store more KV cache, a new block is allocated. A block contains 16 tokens by default. The KV cache memory consumption per token is calculated using the following equation.

head\_size \* num\_kv\_heads \* num\_layers \* dtype\_size \* 2

With the knowledge of all available memory and the size of a token, the vLLM can determine the number of KV cache tokens that can be stored.

The cache engine calls *torch.empty* to create a giant tensor to reserve a huge chunk of memory for the KV cache.

The final step is to initialize the scheduler.

The scheduler creates a block space manager to manage the mapping between the logical KV cache IDs and their physical storing locations. It allocates and swaps the memory. The scheduler also creates three queues, the running, the waiting, and the swapped. More on the scheduler later.

### The Main Loop of the Engine

The engine is always in a loop. It routinely checks if there’s any request in the queue. If so, it executes the model for a step and the loop continues. If not, it idly waits for the arrival of new requests. Check out the code below.

```
async def run_engine_loop(self):  
    has_requests_in_progress = False  
    while True:  
        if not has_requests_in_progress:  
            logger.debug("Waiting for new requests...")  
            await self._request_tracker.wait_for_new_requests()  
            logger.debug("Got new requests!")  
​  
        # Abort if iteration takes too long due to unrecoverable errors  
        # (eg. NCCL timeouts).  
        try:  
            has_requests_in_progress = await asyncio.wait_for(  
                self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)  
        except asyncio.TimeoutError as exc:  
            logger.error(  
                "Engine iteration timed out. This should never happen!")  
            self.set_errored(exc)  
            raise  
        await asyncio.sleep(0)
```

So, what exactly is a *step*? A step is the smallest time slice in vLLM scheduling. It is either to generate a new token or to process new prompts.

Let’s see what the engine wrapper does in a step.

```
async def engine_step(self) -> bool:  
    new_requests, finished_requests = (  
        self._request_tracker.get_new_and_finished_requests())  
​  
    for new_request in new_requests:  
        # Add the request into the vLLM engine's waiting queue.  
        # TODO: Maybe add add_request_batch to reduce Ray overhead  
        try:  
            if self.engine_use_ray:  
                await self.engine.add_request.remote(**new_request)  
            else:  
                await self.engine.add_request_async(**new_request)  
        except ValueError as e:  
            # TODO: use a vLLM specific error for failed validation  
            self._request_tracker.process_exception(  
                new_request["request_id"],  
                e,  
                verbose=self.log_requests,  
            )  
​  
    if finished_requests:  
        await self._engine_abort(finished_requests)  
​  
    if self.engine_use_ray:  
        request_outputs = await self.engine.step.remote()  
    else:  
        request_outputs = await self.engine.step_async()  
​  
    # Put the outputs into the corresponding streams.  
    for request_output in request_outputs:  
        self._request_tracker.process_request_output(  
            request_output, verbose=self.log_requests)  
​  
    return len(request_outputs) > 0
```

It picks up new requests from the tracker queue. Then it calls the engine to add new requests. The engine will encode the prompt, create a sequence group for the request, and inform the scheduler that there’s a new request.

A **sequence group** in the vLLM holds all sequences related to a request. If we use greedy sampling, then a sequence group has only one sequence all the time. However, if we employ beam search, a sequence group may contain multiple sequences.

Then, it calls the engine to move forward one step. The *step* method of the engine is shown below. It first lets the scheduler decide which requests (sequence groups) to run in this iteration. Then it passes the scheduler decision to the workers and let them execute the model on the GPUs. Finally it collects all the output and does some post-processing.

```
def step(self) -> List[RequestOutput]:  
  
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()  
  
    if not scheduler_outputs.is_empty():  
        output = self.model_executor.execute_model(  
            seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,  
            scheduler_outputs.blocks_to_swap_out,  
            scheduler_outputs.blocks_to_copy)  
    else:  
        output = []  
  
    return self._process_model_outputs(output, scheduler_outputs)
```

We will discuss what *scheduler.schedule()* does in the next section.

### Scheduling

The vLLM can do only one thing in each step, either to pre-fill or to decode. To pre-fill is to run the model with prompts and fill in the KV caches of the prompt tokens. To decode is to generate the next token using the existing KV cache from previous tokens.

The vLLM scheduler decides whether to pre-fill or decode in a step based on two factors: if any requests are swapped to the CPU and if there are any new requests. The vLLM prioritizes swapped requests over new requests, following the FCFS (First Come First Serve) rule.

If any requests were swapped previously due to insufficient memory, the vLLM will try to bring them back as soon as possible, at the price of putting new incoming requests on hold.

If no requests are swapped and there are new requests, the vLLM schedules a pre-fill step for these new requests by calling the *\_schedule\_prefills* method. It picks up as many new requests as possible from the queue until the limit is reached. For example, suppose there are five new requests each containing 10k tokens, and the maximum number of pre-filling tokens is configured to be 25k. In that case, the vLLM will pick up two new requests to run in this step and leave the remaining three in the waiting queue.

One notable aspect is the vLLM scheduler’s strict adherence to the FCFS (First Come First Serve) rule. Consider a scenario with five new requests. They contain 2k, 3k, 30k, 2k, and 3k tokens, respectively, and the limit is 25k. The vLLM will pick up only the first two requests from the queue although it can actually run four requests if it picks up both the first two requests and the last two requests. What happens is that it sees the third request, finds it too large, says no and stops, not going to assess subsequent requests in the queue.

If there are no new requests, or if there are swapped requests, the vLLM schedules a decoding step. There are two types of requests to decode. One is requests in the GPU memory. The other one is victim requests swapped to the CPU memory as we mentioned above.

You might wonder in what scenario would requests be swapped. For example, we run 5 requests with 1k prompt tokens each and the size of the KV cache pool is 6k tokens. At first, all 5 requests stay in the GPU memory. However, after all of them have generated 200 tokens, the size of their total KV cache comes to 6k, which reaches the limit of the KV cache pool. There is no space left for any newly generated tokens. The vLLM solves this problem by moving the lowest priority request from the GPU memory to the CPU memory. Then an amount of space for 1200 tokens is freed and the remaining 4 requests can continue to generate new tokens. The victim request will be waiting in the CPU memory for any of the 4 requests to finish. Once a running request is finished the memory it takes can be freed, and we can bring the victim request back to the running queue.

Another intriguing aspect concerns the handling of victim requests. The straightforward approach involves transferring the KV cache tensor of the victim request from GPU to CPU, and sending it back, resulting in significant communication overhead. The size of the tensor can be hundreds of megabytes. Remember communication is costly and it frequently serves as the bottleneck for inference efficiency in modern LLMs. However, the vLLM cleverly solves this problem. Rather than shuttling the hefty KV cache tensors across devices, it just drops them and remembers the generated tokens instead. When the victim request is brought back, the vLLM simply concatenates the generated tokens with the prompt tokens, treats the request as a new request, and recomputes all its KV cache. For example, if the prompt is “San Francisco is”, and “a city in” is generated before the request is out, the vLLM will simply update the prompt to “San Francisco is a city in” and put the request to the new requests queue. Because computation in the pre-fill phase is very fast, faster than communication, this recompute strategy leads to better performance. Note that, however, as of the time of writing, the recompute strategy is not compatible with beam search. So, if beam search is employed, you will fall to the naive swap strategy which transfers the entire chunky KV cache tensor back and forth.

In a decoding step, the scheduler first checks if there’s enough space in the GPU KV cache pool to accommodate all new tokens generated in this step. If not, it means at least one low-priority request should be moved out of the GPU and the memory of its KV cache freed. The remaining requests are scheduled. On the other hand, if there are plenty room for requests in the GPU to generate new tokens, the scheduler will look into the swapped requests queue to see if it can bring back any previously swapped victim requests.

The scheduler returns its decision including whether to pre-fill or decode, which requests to schedule, which KV cache blocks to swap, and others. The vLLM passes the *SchedulerOutput* to the workers and they will arrange the KV cache and execute the model according to these instructions.

```
async def step_async(self) -> List[RequestOutput]:  
     
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()  
  
    if not scheduler_outputs.is_empty():  
        # Execute the model.  
        output = await self.model_executor.execute_model_async(  
            seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,  
            scheduler_outputs.blocks_to_swap_out,  
            scheduler_outputs.blocks_to_copy)  
    else:  
        output = []  
  
    return self._process_model_outputs(output, scheduler_outputs)
```

### Conclusion

I hope this article offers a quick understanding of how the vLLM works. However, there remain untouched aspects such as the implementation of PagedAttention, the model execution pipeline, and the inference of quantized models in the vLLM. Besides, the vLLM community is actively developing new features like chunked pre-fills and speculative inference. Stay tuned, we plan to delve into these topics in future articles.
