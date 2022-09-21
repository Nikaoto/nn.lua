local nn = require("nn")
--local training_data, testing_data = require("data")

math.randomseed(6929)

local net = nn.new_neural_net({
   neuron_counts = {1, 4, 2},
   act_fns = {nil, sigmoid, sigmoid},
   d_act_fns = {nil, d_sigmoid, d_sigmoid},
})

nn.train(net, training_data, {
   epochs = 1000,
   learning_rate = 0.1,
   log_freq = 0.5,
})

for i, data in ipairs(testing_data) do
   local out = nn.feedforward(net, {
      inputs = data.inputs
   })
   print(string.format([[
test %i
data: %s
prediction: %s
-----]], i, inspect(data), inspect(out)))
end
