# Efficient GPU Utilization
- [1. CUDA out of memory solutions ](#1-cuda-out-of-memory-solutions)
  - [1.1. Use a smaller batch size](#11-use-a-smaller-batch-size)
  - [1.2. Check if there is any accumulated history across your training loop](#12-check-if-there-is-any-accumulated-history-across-your-training-loop)
  - [1.3 Delete intermediate variables you don't need](#13-delete-intermediate-variables-you-dont-need)
  - [1.4. Check if you GPU memory is freed properly](#14-check-if-you-gpu-memory-is-freed-properly)
  - [1.5. Turn off gradient calculation during validation](#15-turn-off-gradient-calculation-during-validation)
  - [1.6. COM in Google Colab](#16-com-in-google-colab)
- [2. Multiple GPUs](#2-multiple-gpus)

## 1. CUDA out of memory solutions

<div align=center>
  <img src='images/COM.jpg' width=360 height=240>
</div>

- Anyone engaged in deep learning must have encountered the problem of cuda out of memory.Sometimes It's really frustrating when you've finished writing the code and debugged it for a week to make sure everything is correct. Just when you start training, the program throws a `CUDA out of memory` error. Here are some practical ways to hepl you solve this annoying problem
### 1.1. Use a smaller batch size
- The most frequent cause of this problem is that your batch size is set too large. Try to use a small one.
- In some special scenarios, a smaller batch size may cause your network performance to drop, so a good way to balance this is to use gradient accumulation. Here is an example
    ```python
    accumulation_steps = 10                                                              # Reset gradients tensors
    for i, (inputs, labels) in enumerate(training_set):
        predictions = model(inputs)                     # Forward pass
        loss = loss_function(predictions, labels)       # Compute loss function
        loss = loss / accumulation_steps                # Normalize our loss (if averaged)
        loss.backward()                                 # Backward pass
        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            model.zero_grad()                           # Reset gradients tensors
            if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
                evaluate_model()                        # ...have no gradients accumulate
    ```
- As you can see from the code, `model.zero_grad()` is executed only after the forward count reaches `accmulation_step`, i.e. the gradient is accumulated 10 times before updating the parameters. This allows you to have a relatively large batch size while reducing the memory footprint.
- This may also have some minor problems, such as the BN layer may not be calculated accurately, etc.

### 1.2. Check if there is any accumulated history across your training loop
- By default, computations involving variables that require gradients will keep history. This means that you should avoid using such variables in computations which will live beyond your training loops, e.g., when tracking statistics. Instead, you should detach the variable or access its underlying data.
- Here is a bad example:
    ```python
    total_loss = 0
    for i in range(10000):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output)
        loss.backward()
        optimizer.step()
        total_loss += loss
    ```
- `total_loss` is defined outside the loop and will keep accumulating in each loop. This can lead to unnecessary memory usage and you can solve it in two ways: use `total_loss += loss.detach()` or `total_loss += loss.item()` instead.

### 1.3 Delete intermediate variables you don't need
- If you assign a Tensor or Variable to a local, Python will not deallocate until the local goes out of scope. You can free this reference by using del x. Similarly, if you assign a Tensor or Variable to a member variable of an object, it will not deallocate until the object goes out of scope. You will get the best memory usage if you don’t hold onto temporaries you don’t need.
```python
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
output = h(result)
return output
```
- Here, intermediate remains live even while h is executing, because its scope extrudes past the end of the loop. To free it earlier, you should `del intermediate` when you are done with it.

### 1.4. Check if you GPU memory is freed properly
- Sometimes even if your code stops running, the video memory may still be occupied by it. The best way is to find the process engaging gpu memory and kill it
- find the PID of python process from:
    ```bash
    nvidia-smi
    ```
- copy the PID and kill it by:
    ```bash
    sudo kill -9 pid
    ```

### 1.5. Turn off gradient calculation during validation
- You don't need to calculate gradients for forward and backward phase during validation.
    ```python
    with torch.no_grad():
        for batch in loader:
            model.evaluate(batch)
    ```

### 1.6. COM in Google Colab
- If you are getting this error in Google Colab, then try this
    ```python
    import torch
    torch.cuda.empty_cache()
    ```

## 2. Multiple GPUs
