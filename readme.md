# nn.lua
Small neural network library in a single lua file.
Good for learning neural networks & playing with machine learning.

## Usage
Simply copy `nn.lua` into your project and require it.

`examples/learn_xor.lua` is a small but complete example of typical usage, which is:
```lua
-- Rrequire the module
local nn = require("nn")

-- Initialize the network
local net = nn.new_neural_net({
   neuron_counts = {2, 4, 1},
   -- TODO: the act_fns don't work this way yet
   act_fns = {nil, "sigmoid", nil}
})

-- Train it
nn.train(net, training_data, {
   epochs = 1000,
   learning_rate = 0.1
})

-- Use it
local outputs = nn.feedforward(net, {inputs={10, 20}})
print(outputs[1])
```

## TODO
- make `act_fns` a string array (use `raw_act_fns` and `raw_d_act_fns` to hold
  the lua functions internally)
- write a description of how to run the visual approximator w/ controls
- write nn.lua documentation
- make `act_fns` array have the same count as the `neurons` array, instead of
  `act_fns` starting from the second neuron layer. The `biases` array works this
  way already.
- `nn.load_neural_network()`/`nn.save_neural_network()` functions
- get MNIST to work
- Make a GAN
- Potential optimizations include:
  - rewrite in C
  - use SIMD
  - use matrix multiplication w/ video card instead of cartesian nested loop
- Solve a rubik's cube

## Screenshots
Approximating a sine wave in the range [-10, 10] using a single 100-neuron
hidden layer using a ReLU activation function. The training data is only 100
random samples from the sine wave from the given range.

After approximately one hour of training:
![Sine wave after 1 hour of training](./screenshots/nn-screenshot-approx-sin-relu-1.png)

Notice how the network doesn't care to fit anything beyond the range [-10, 10]
as it doesn't have the data for it

A close up:
![Close up sine wave after 1 hour of training](./screenshots/nn-screenshot-approx-sin-relu-2.png)

When the same is done using a sigmoid function instead, the results are much
better.

After only 5 minutes:
![Sine wave after 3 minutes of training using sigmoid](./screenshots/nn-screenshot-approx-sin-sigmoid.png)

## License (3-clause BSD)
```
Copyright 2022 Nikoloz Otiashvili.

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials
   provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products
   derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
