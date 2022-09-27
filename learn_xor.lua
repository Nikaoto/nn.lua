local inspect = require("lib/inspect")
local nn = require("nn")

-- Training data (XOR function)
local training_data = {
   { inputs={1,1}, outputs={0} },
   { inputs={1,0}, outputs={1} },
   { inputs={0,1}, outputs={1} },
   { inputs={0,0}, outputs={0} }
}
local testing_data = training_data

-------------
print("\nNet")
math.randomseed(1337)
local net = nn.new_neural_net({
   neuron_counts = {2, 4, 1},
   act_fns = {nn.sigmoid},
   d_act_fns = {nn.d_sigmoid},
})
print(inspect(net))

------------------
print("\nTraining")
nn.train(net, training_data, {
   epochs = 1000,
   learning_rate = 0.1,
   log_freq = 0.005,
})

------------------
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
