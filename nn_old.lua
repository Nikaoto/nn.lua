-- nn
require("util")
local inspect = require("inspect")
local fmt = string.format

function sum_layer(arr)
   local sum = 0
   for i, _ in ipairs(arr) do
      sum = sum + arr[i]
   end
   return sum
end

function run_model(net, act_fn)
   -- Input layer sum
   local prv_sum = sum_layer(net[1])

   -- Go thru hidden layers and count sums
   local tmp_layer
   local layer_idx = 2
   while layer_idx <= #net do
      tmp_layer = {}
      for wi, _ in ipairs(net[layer_idx]) do
         local activation = act_fn(
            net[layer_idx][wi] * prv_sum)
         table.insert(tmp_layer, activation)
      end

      prv_sum = sum_layer(tmp_layer)
      layer_idx = layer_idx + 1
   end

   -- Output layer is in tmp_layer
   return tmp_layer
end

-- Insert inputs into first layer of neural net
function insert_inputs(net, inputs)
   for i, _ in ipairs(net[1]) do
      net[1][i] = inputs[i]
   end
end

-- Calculate the loss of a single output layer
function calc_loss(actual, desired)
   local loss = 0
   for i, _ in ipairs(actual) do
      loss = loss + sq(actual[i] - desired[i])
   end
   return loss
end

-- Returns training data shaped like:
-- {
--   { inputs: {...}, outputs: {...} },
--   { inputs: {...}, outputs: {...} },
--   ...
-- }
function generate_training_data(fn, n, seed)
   local data = {}
   math.randomseed(seed or os.time())
   for i=1, n do
      local a = math.random(-50, 50)
      local ans = fn(a)
      data[i] = { inputs={a, b}, outputs={ans}}
   end
   return data
end

function main()
   -- Init neural net
   local fn_to_discover = function(a)
      return a*a
   end
   local diff_step = 0.0001
   local learning_rate = 0.000000000001
   local input_layer = {1}
   local hidden_layer_1 = rand_arr(-5, 5, 4, 123)
   local hidden_layer_2 = rand_arr(-5, 5, 4, 456)
   --local hidden_layer_3 = rand_arr(-5, 5, 4,  789)
   local output_layer = {3}
   local iterations = 16000
   local print_percent = 0.01
   local net = {
      input_layer,
      hidden_layer_1,
      hidden_layer_2,
      --hidden_layer_3,
      output_layer
   }

   -- Activation functions to choose from: relu, linear
   local activation_fn = linear

   print("\n------- Initial neural net:")
   print(inspect(net))

   -- Init testing/training data
   local training_data = generate_training_data(
      fn_to_discover, 100, 87)
   local testing_data = generate_training_data(
      fn_to_discover, 3, 123)

   --[[ Training loop ]]--
   local i = 1
   while i <= iterations do
      -- Calculate net loss and gradients of each weight
      local total_loss = 0
      local avg_loss = 0
      local step = diff_step
      local grad_mat = {}

      shuffle(training_data)
      --local j = math.random(1, #training_data)
      for j, _ in ipairs(training_data) do
         insert_inputs(net, training_data[j].inputs)
         local results = run_model(net, activation_fn)
         local l1 = calc_loss(results, training_data[j].outputs)
         total_loss = total_loss + l1

         -- Last hidden layer loss
         -- local loss_vec = {}
         -- local li = #net - 1
         -- loss_vec[li] = 0
         -- if not grad_mat[li] then
         --    grad_mat[li] = map(net[li], function(w) return 0 end)
         -- end
         -- for wi, _ in ipairs(net[li]) do
         --    local w1, w2 = net[li][wi], net[li][wi] + step
         --    net[li][wi] = w2
         --    local l2 = calc_loss(run_model(net, activation_fn),
         --                         training_data[j].outputs)
         --    local grad = (l2 - l1) / (w2 - w1)
         --    net[li][wi] = w1
         --    grad_mat[li][wi] = grad_mat[li][wi] + grad / #training_data
         --    loss_vec[li] = loss_vec[li] + l2
         -- end

         for li=(#net-1), 2, -1 do
            if not grad_mat[li] then
               grad_mat[li] = map(net[li], function(w) return 0 end)
            end

            for wi, _ in ipairs(net[li]) do
               if not grad_mat[li][wi] then
                  grad_mat[li][wi] = 0
               end

               local w1 = net[li][wi]
               local w2 = w1 + step
               net[li][wi] = w2
               local r = run_model(net, activation_fn)
               local l2 = calc_loss(r, training_data[j].outputs)
               local grad = (l2 - l1) / (w2 - w1)
               grad_mat[li][wi] = grad_mat[li][wi] + grad
               net[li][wi] = w1
            end
         end
      end
      avg_loss = total_loss / #training_data

      if i % (iterations * print_percent) == 0 then
         print(string.format("iteration = %i, avg_loss = %f",
            i, avg_loss))

         -- print(string.format("iteration = %i, loss = %f",
         --    i, l1))
      end

      -- Apply nudges to weights
      for li=(#net - 1), 2, -1 do
         for wi, _ in ipairs(net[li]) do
            local nudge = grad_mat[li][wi] * learning_rate
            net[li][wi] = net[li][wi] - nudge
         end
      end

      i = i + 1
   end

   -- Run model against testing data
   print("\n-------- Running against testing data")
   for j, _ in ipairs(testing_data) do
      print(string.format("test %i: %s", j, inspect(testing_data[j])))
      insert_inputs(net, testing_data[j].inputs)
      print(string.format("result %i: %s", j, inspect(run_model(net, activation_fn))))
   end

   print("\n-------- Final neural net:")
   print(inspect(net))
end

main()
