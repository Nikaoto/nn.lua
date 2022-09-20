local nn = require("nn")
local inspect = require("inspect")
local fmt = string.format
require("util")

local function randf(min, max)
   return lerp(min, max, math.random(0,10000) / 10000)
end

local function generate_training_data(fn, n, seed)
   local data = {}
   math.randomseed(seed or os.time())
   for i=1, n do
      local a = randf(0.1, 1.5)
      local ans = fn(a)
      data[i] = { inputs={a}, outputs={ans}}
   end
   return data
end

local fn_to_discover = function(a)
   return math.sin(a)
end

print("\n-------- Initial neural net")
local net = nn.new_neural_net({
   neuron_counts = {1, 6, 1},
   act_fns = {nil, relu, nil},
   d_act_fns = {nil, d_relu, nil},
})
print(inspect(net))

-- Generate training & testing data
local training_data = generate_training_data(
   fn_to_discover, 300, 123)
local testing_data = generate_training_data(
   fn_to_discover, 3, 456)

-- Train
nn.train(net, training_data, {
   shuffle = true,
   epochs = 3500,
   learning_rate = 0.0000001,
   log_freq = 0.01
})

print("\n-------- Final neural net")
print(inspect(net))

-- Test
print("\n-------- Running against testing data")
for j, _ in ipairs(testing_data) do
   print(fmt("test %i: %s", j, inspect(testing_data[j])))
   local out = nn.feedforward(net, {
      inputs=testing_data[j].inputs
   })
   print(fmt("result %i: %s", j, inspect(out)))
end
