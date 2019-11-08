# Autograd: Automatic Differenciation

## The autograd package provides automatic differentiation for all operations on Tensors.
## It is a define-by-run framework, which means that backprop is defined by how code is run
## and that every single iteration can be different.

## set requires_grad as True, it starts to track all operations on it. 
## After finishing computation call .backward() and have all the gradients computed automatically.
## The gradient for this tensor will be accumulated into .grad attribute.

## .detach() : to stop tracking (as well as using memory for tracking)
## wrap the code block in with torch.no_grad(): This can be particularly helpful when evaluating a model because the model may have trainable parameters with requires_grad=True, but for which we don’t need the gradients.


import torch
mat = torch.rand(2, 2, requires_grad = True)
temp = torch.ones(2, 2)
mat = mat + temp
#mat = mat * 10
print("Printing tensor with tracking information(after summetion):\n", mat)
mat = mat * 10
print("Printing tensor with tracking information(after multiplication):\n", mat)


mother = torch.tensor([20.0, 40.0, 80.0], requires_grad=True)
child = mother * 2
print("Dataflow\n")
while child.data.norm() < 1000.0 :
  print("before multiplication: ",child.data.norm)
  child = child * 2
  print("after multiplication: ", child)
print("\n\n")
print("Final result: ", child)

newChild = torch.tensor([2.0, 4.0, 5.0], dtype=torch.float)
child.backward(newChild)

print(newChild)
print("Final Backward result to the mother: ", mother.grad)