local nn = {
   _VERSION = "nn.lua 0.0.1",
   _URL     = "http://github.com/Nikaoto/nn.lua",
   _DESCRIPTION = "small neural network library",
   _LICENSE = [[ BSD3 ]]
}

-- Defaults
local def = {
   act_fns = {},
   d_act_fns = {},
   learning_rate = 0.001,
   epochs = 1000,
   training_log_freq = 0.01,
   weight_min = -5,
   weight_max = 5,
   bias_min = -5,
   bias_max = 5,
   shuffle_training_data = false,
   randomize_weights = true,
   randomize_biases = true,
}

local fmt = string.format

local function sq(x) return x*x end

local function lerp(a, b, p) return a + (b-a)*p end

local function rand_lerp(min, max)
   return lerp(min, max, math.random(0, 1000) / 1000)
end

local function map(arr, fn)
    local new_arr = {}    
    for i, _ in ipairs(arr) do
        new_arr[i] = fn(arr[i], i)
    end
    return new_arr
end

-- Make an array of 'nmemb' elements and set all of them to 'value'
local function make_array(nmemb, value)
   local arr = {}
   for i=1, nmemb do table.insert(arr, value) end
   return arr
end

-- Calculate the error for a single example
local function calc_err(actual, desired)
   return sq(actual - desired)
end

-- Calculate the derivative of the error function
local function calc_d_err(actual, desired)
   return 2 * (actual - desired)
end

-- Calculate the sum of all errors
local function calc_total_err(actual_arr, desired_arr)
   local loss = 0
   for i, _ in ipairs(actual_arr) do
      loss = loss + calc_err(actual_arr[i], desired_arr[i])
   end
   return loss
end

-- Returns a table that represents a neural network
function nn.new_neural_net(opts)
   if not opts then opts = {} end
   local neuron_counts = opts.neuron_counts
   local act_fns       = opts.act_fns    or def.act_fns
   local d_act_fns     = opts.d_act_fns  or def.d_act_fns
   local weight_min    = opts.weight_min or def.weight_min
   local weight_max    = opts.weight_max or def.weight_max
   local bias_min      = opts.bias_min   or def.bias_min
   local bias_max      = opts.bias_max   or def.bias_max
   local weights       = opts.weights
   local biases        = opts.biases

   if not neuron_counts then
      error("nn.new_neural_net(): no opts.neuron_counts given")
   end

   local rand_weight = function()
      return rand_lerp(weight_min, weight_max)
   end

   local rand_bias = function()
      return rand_lerp(bias_min, bias_max)
   end

   local net = {
      act_fns = act_fns,
      d_act_fns = d_act_fns,
      neurons = map(neuron_counts, make_array),
   }

   --[[ NOTE:
      Weights are defined left-to-right.
      Meaning, in a net like this:
      +-----------------+
      | in  hidden  out |
      | --  ------  --- |
      |      c          |
      | a    d  g    i  |
      | b    e  h    k  |
      |      f          |
      +-----------------+
      The weights are defined as:
      {
        { ac, ad, ae, af, bc, bd, be, bf },
        { cg, ch, dh, dh, eg, eh, fg, fh },
        { gi, gk, hi, hk }
      }

      The reason behind this is that it becomes easier to index
      the weights relative to the neuron indices.
   ]]--

   -- Create random weights if none given
   if not weights then
      weights = {}
      for li=1, #net.neurons-1 do
         weights[li] = {}
         for i=1, #net.neurons[li] * #net.neurons[li+1] do
            table.insert(weights[li], rand_weight())
         end
      end
   end
   net.weights = weights

   -- Create random biases if none given
   if not biases then
      biases = map(net.neurons, function(layer)
         return map(layer, rand_bias)
      end)
   end
   net.biases = biases

   return net
end

function nn.train(net, opts)
   if not opts then opts = {} end
   -- TODO: read opts
   -- TODO: training loop, accumulate nudges using backprop and
   --       print status
   -- TODO: apply nudges
end

function nn.insert_inputs(net, inputs)
   for i, _ in ipairs(net.neurons[1]) do
      net.neurons[1][i] = inputs[i]
   end
end

-- Feeds inputs forward and returns output layer
function nn.feedforward(net, opts)
   if not opts then opts = {} end
   if opts.inputs then nn.insert_inputs(net, opts.inputs) end

   for li=2, #net.neurons, 1 do
      for ni, _ in ipairs(net.neurons[li]) do
         local sum = 0

         -- Dot product of previous layer's neurons and weights
         for plni, _ in ipairs(net.neurons[li-1]) do
            -- 'plni' stands for 'previous layer neuron index'
            local wi = ni + (plni-1) * #net.neurons[li]
            local weight = net.weights[li-1][wi]
            local prev_activation = net.neurons[li-1][plni]
            sum = sum + prev_activation * weight
         end

         -- Add bias
         local act = sum + net.biases[li][ni]

         -- Apply activation function
         if net.act_fns and net.act_fns[li] then
            net.neurons[li][ni] = net.act_fns[li](act)
         else
            net.neurons[li][ni] = act
         end
      end
   end

   return net.neurons[#net.neurons]
end


return nn
