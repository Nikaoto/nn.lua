inspect = require("inspect")
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
math.randomseed(6929)
local net = nn.new_neural_net({
   neuron_counts = {2, 3, 1},
   act_fns = {nn.sigmoid, nn.sigmoid},
   d_act_fns = {nn.d_sigmoid, nn.d_sigmoid},
   biases = {
      {2.8, 2.8},
      {-4.31,-4.31,-4.31},
      {-0.68},
   }
})
print(inspect(net))

------------------
print("\nTraining")
nn.train(net, training_data, {
   epochs = 2500,
   learning_rate = 0.1,
   log_freq = 0.01,
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
