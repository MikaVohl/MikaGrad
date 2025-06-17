# MikaGrad

> MikaGrad is my implementation of an autograd engine mimicking Pytorch or Karpathy's Micrograd


## Components of an autograd engine
- Value class which implements methods for algebraic operations and functions
- Each value object can generate a gradient
- Value objects point to their children (the values involved in creating self)
- 


## Future support 
- Vector values / tensor values