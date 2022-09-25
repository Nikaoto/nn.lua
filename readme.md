# nn.lua
Small neural network library for Lua.

## TODO
- make `act_fns` array have the same count as the `neurons` array, instead of
  `act_fns` starting from the second neuron layer. The `biases` array works this
  way already.
- `nn.load_neural_network()`/`nn.save_neural_network()` functions
- get MNIST to work
- Approximate a sine wave (Andrew Ng course may have info)
- Make a GAN
- Potential optimizations include:
  - rewrite in C
  - use SIMD
  - use matrix multiplication w/ video card instead of cartesian nested loop
- Solve a rubik's cube
