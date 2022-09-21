local nn = {}

-- Defaults
local def = {
   act_fns = {},
   d_act_fns = {},
   diff_step = 0.001,
   learning_rate = 0.001,
   epochs = 10000,
   shuffle_training_data = false,
   training_log_freq = 0.01,
   weight_min = -5,
   weight_max = 5
}

local fmt = string.format

local function sq(x) return x*x end

local function map(arr, fn)
   local new_arr = {}
   for i, _ in ipairs(arr) do
      new_arr[i] = fn(arr[i], i)
   end
   return new_arr
end

local function calloc_tbl(n, data)
   local tbl = {}
   for i=1, n do
      table.insert(tbl, data)
   end
   return tbl
end

local function calc_loss(actual, desired)
   local loss = 0
   for i, _ in ipairs(actual) do
      loss = loss + sq(actual[i] - desired[i])
   end
   return loss
end

local function calc_d_loss(actual, desired)
   return (actual - desired) * 2
end

-- Shuffle array in place
local function shuffle(arr)
   for i=#arr, 1, -1 do
      local j = math.random(1, i)
      arr[j], arr[i] = arr[i], arr[j]
   end
end

local function lerp(a, b, p)
   return a + (b-a)*p
end

local function rand_weight(min, max)
   return lerp(min, max, math.random(0, 1000) / 1000)
end

function nn.new_neural_net(opts)
   if not opts.neuron_counts then return nil end
   local neuron_counts = opts.neuron_counts
   local seed          = opts.seed       or os.time()
   local weight_min    = opts.weight_min or def.weight_min
   local weight_max    = opts.weight_max or def.weight_max
   local act_fns       = opts.act_fns    or def.act_fns
   local d_act_fns     = opts.d_act_fns  or def.d_act_fns
   local rw = function()
      return rand_weight(weight_min, weight_max)
   end
   local rb = rw

   math.randomseed(seed)

   -- Create net with its neurons
   local net = {
      act_fns = act_fns,
      d_act_fns = d_act_fns,
      neurons = map(neuron_counts, function(c, i)
         return calloc_tbl(c, i)
      end)
   }

   -- Create weights
   net.weights = {}
   for li=1, #net.neurons-1 do
      net.weights[li] = {}
      for i=1, #net.neurons[li] * #net.neurons[li+1] do
         table.insert(net.weights[li], rw())
      end
   end

   -- Create biases
   net.biases = map(net.neurons, function(layer)
      return map(layer, function(neuron) return rb() end)
   end)

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

   return net
end

function nn.insert_inputs(net, inputs)
   if #net.neurons < 1 then error("invalid net") end
   if #inputs ~= #net.neurons[1] then
      error("#inputs ~= #net.neurons[1]")
   end

   for neuron_idx, _ in ipairs(net.neurons[1]) do
      net.neurons[1][neuron_idx] = inputs[neuron_idx]
   end
end


-- Feeds forward the inputs and returns output layer
function nn.feedforward(net, opts)
   if opts.inputs then
      nn.insert_inputs(net, opts.inputs)
   end

   for li=2, #net.neurons, 1 do
      for ni, _ in ipairs(net.neurons[li]) do
         local sum = 0

         -- 'plni' stands for 'previous layer neuron index'
         for plni, _ in ipairs(net.neurons[li-1]) do
            local wi = ni + (plni-1) * #net.neurons[li]
            local weight = net.weights[li-1][wi]
            local activation = net.neurons[li-1][plni]
            sum = sum +  activation * weight
         end

         -- Apply activation function
         local act = sum + net.biases[li][ni]
         if net.act_fns and net.act_fns[li] then
            net.neurons[li][ni] = net.act_fns[li](act)
         else
            net.neurons[li][ni] = act
         end
      end
   end

   return net.neurons[#net.neurons]
end

function nn.train(net, training_data, opts)
   if not opts then opts = {} end
   local learning_rate = opts.learning_rate or def.learning_rate
   local step          = opts.diff_step or def.diff_step
   local epochs        = opts.epochs or def.epochs
   local shuf          = opts.shuffle or def.shuffle_training_data
   local log_freq      = opts.log_freq or def.training_log_freq
   local log_every     = 1 / log_freq

   for iter=1, epochs do
      local derivs = {}
      local grad_mat = {}
      local bias_grad_mat = {}
      local avg_loss = 0

      if shuf then shuffle(training_data) end
      for i, _ in ipairs(training_data) do
         local out1 = nn.feedforward(net, {
            inputs = training_data[i].inputs })
         local loss1 = calc_loss(out1, training_data[i].outputs)
         avg_loss = avg_loss + loss1 / #training_data

         -- Calculate grad for the last weight layer (hidden->output)
         --[[local li = #net.weights
         if not grad_mat[li] then grad_mat[li] = {} end
         if not bias_grad_mat[li] then bias_grad_mat[li] = {} end
         if not derivs[li] then derivs[li] = {} end

         for oni, onv in ipairs(net.neurons[li+1]) do
            local d_loss = calc_d_loss(onv, training_data[i].outputs[oni])
            local d_act = net.d_act_fns[li+1] and net.d_act_fns[li+1](onv) or 1
            for wi=oni, #net.weights[li], #net.neurons[li+1] do
               local hni = math.ceil(wi / #net.neurons[li+1])
               derivs[li][hni] = d_loss * d_act * net.weights[li][wi] +
                  (derivs[li][hni] or 0)
               grad_mat[li][wi] = d_loss * d_act * net.neurons[li][hni] +
                  (grad_mat[li][wi] or 0)
               bias_grad_mat[li][hni] = d_loss * d_act +
                  (bias_grad_mat[li][hni] or 0)
            end
         end

         -- Calculate grad for the other hidden layers
         for li=#net.weights-1, 1, -1 do
            if not grad_mat[li] then grad_mat[li] = {} end
            if not bias_grad_mat[li] then bias_grad_mat[li] = {} end
            if not derivs[li] then derivs[li] = {} end

            for di, dv in ipairs(derivs[li+1]) do
               local d_act = net.d_act_fns[li+1] and net.d_act_fns[li+1](dv) or 1
               for wi=di, #net.weights[li], #net.neurons[li+1] do
                  local hni = math.ceil(wi / #net.neurons[li+1])
                  derivs[li][hni] = d_act * net.weights[li][wi] +
                     (derivs[li][hni] or 0)
                  grad_mat[li][wi] = d_act * net.neurons[li][hni] +
                     (grad_mat[li][wi] or 0)
                  bias_grad_mat[li][hni] = d_act +
                     (bias_grad_mat[li][hni] or 0)
               end
            end
         end]]--

         -- [[ CALCULATE USING FORWARD PROP ]]--
         -- Biases
         for li, _ in ipairs(net.biases) do
            if not bias_grad_mat[li] then bias_grad_mat[li] = {} end

            for bi, _ in ipairs(net.biases[li]) do
               -- Tune bias
               local b1 = net.biases[li][bi]
               local b2 = b1 + step
               net.biases[li][bi] = b2
   
               -- Run model
               local out2 = nn.feedforward(net, {
                  inputs = training_data[i].inputs })
   
               -- Calculate difference in losses of the two runs
               local loss2 = calc_loss(out2, training_data[i].outputs)
               local grad = (loss2-loss1) / step
   
               -- Reset bias
               net.biases[li][bi] = b1
   
               -- Accumulate gradient
               bias_grad_mat[li][bi] = (bias_grad_mat[li][bi] or 0) + grad
            end
         end

         -- Weights
         for li, _ in ipairs(net.weights) do
            if not grad_mat[li] then grad_mat[li] = {} end

            for wi, _ in ipairs(net.weights[li]) do
               -- Tune weight
               local w1 = net.weights[li][wi]
               local w2 = w1 + step
               net.weights[li][wi] = w2

               -- Run model
               local out2 = nn.feedforward(net, {
                  inputs = training_data[i].inputs })

               -- Calculate difference in losses of the two runs
               local loss2 = calc_loss(out2,
                  training_data[i].outputs)
               local grad = (loss2-loss1) / step

               -- Reset the weight
               net.weights[li][wi] = w1

               -- Accumulate gradients in grad_mat
               grad_mat[li][wi] = (grad_mat[li][wi] or 0) + grad
            end
         end
      end

      -- Log status
      if iter % log_every == 0 then
         print(fmt("epoch = %i, avg_loss = %f", iter, avg_loss))
      end

      -- Apply nudges to weights
      for li, _ in ipairs(net.weights) do
         for wi, _ in ipairs(net.weights[li]) do
            local nudge = grad_mat[li][wi] * learning_rate
            net.weights[li][wi] = net.weights[li][wi] - nudge
         end
      end

      -- Apply nudges to biases
      for li, _ in ipairs(net.biases) do
         for bi, _ in ipairs(net.biases[li]) do
            local nudge = bias_grad_mat[li][bi] * learning_rate * 10
            net.biases[li][bi] = net.biases[li][bi] - nudge
         end
      end
   end
end

return nn
