local inspect = require("lib/inspect")
local nn = require("nn")

local function lerp(a, b, p) return a + (b-a)*p end

local function randf(min, max)
   return lerp(min, max, math.random(0,10000) / 10000)
end

local function generate_training_data(fn, n, seed)
   local data = {}
   for i=1, n do
      local a = randf(0, 4)
      local ans = fn(a)
      data[i] = { inputs={a}, outputs={ans}}
   end
   return data
end

local fn_to_discover = function(a) return math.sin(a) end

local SEED = 1337

-- Init
print("\nNet")
math.randomseed(SEED)
local net = nn.new_neural_net({
   neuron_counts = {1, 12, 1},
   act_fns = {"sigmoid"},
})
print(inspect(net))

-- Generate training & testing data
local training_data = generate_training_data(
   fn_to_discover, 300)
local testing_data = generate_training_data(
   fn_to_discover, 3)

-- Train
print("\nTraining")
nn.train(net, training_data, {
   shuffle = true,
   epochs = 2000,
   learning_rate = 0.01,
   log_freq = 0.01
})
print("\nNet after training")
print(inspect(net))

-- Test
print("\nTesting")
for i, data in ipairs(testing_data) do
   local out = nn.feedforward(net, {
      inputs = data.inputs
   })
   print(string.format([[
-----
test %i: %s
prediction: %s]], i, inspect(data), inspect(out)))
end
