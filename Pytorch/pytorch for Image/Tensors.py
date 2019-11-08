import torch
import numpy as np
# Pytorch tensors can be used on a GPU

## initialize a empty tensor
zero = torch.empty(5, 3)
print("Printing Zero:\n", zero)

## Randomly initiazed Matrix

random = torch.rand(5,3)
print("Printing randomly initialzed matrix:\n", random)

## Matrix filled with zero with long datatype
longDT = torch.ones(5, 3, dtype = torch.long)
print("Priting a long dtype Matrix:\n", longDT)


# constract a tensor Directly from data
x = torch.tensor([[5, 2], [4, 3]])
print("Tensor directly from data: ", x)
y = x.new_ones(2,2)
print("Tensor created from another tensor, a clone: \n", y)

# Printing tensor size
print(x.size())

# adding 2 tensors
result = torch.zeros(2,2)
torch.add(x, y, out = result)
print("Adding 2 tensors:\n", result)
print("Result DataType: ", result.dtype)
### Adding y to x, in that case resulting data type will be x's datatype 
x.add_(y)
print("Added y to x:\n", x)
## or we can do x+y instead of torch.add
print("\n\n\n")

### Resizing Torch
x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1, 4, 2) # clearly -1 for rest 
print("Priniting size after resizing: ",x.size(), y.size(), z.size())
print("\n\n")


### get value as python number 
print("Normal tensor value: ",  x[0][0])
print("Get tensor value as a python Number: ", x[0][0].item())
print("\n\n")


### Converting tensors into numpy array
x = x.numpy()
print("x converted into a numpy array from tensor:\n", x)
### converting back to tensor from numpy
x = torch.from_numpy(x)
print("x converted back to a Tensor:\n", x)

### Tensors can be moved onto any device using the .to method.
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  