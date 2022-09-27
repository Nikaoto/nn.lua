-- Approximate a function using a neural net while graphing both
local graphlove = require("lib/graphlove")
local nn = require("nn")

local WIDTH, HEIGHT = 1024, 720
local lg = love.graphics
local function sq(x) return x*x end
function lerp(a, b, p) return a + (b - a) * p end
function randf(min, max) return lerp(min, max, math.random(0, 1000) / 1000) end

local net, training_data
local graph, actual_curve, predicted_curve
local desired_fn = math.sin

-- Returns a table of points that looks like
-- {x1, y1, x2, y2, x3...}
local function generate_points(x_start, x_end, fn)
   local points = {}
   for x=x_start, x_end, 0.005 do
      table.insert(points, x)
      table.insert(points, fn(x))
   end
   return points
end

function love.conf(t)
   t.console = true
end

function love.load()
   love.window.setMode(WIDTH, HEIGHT)

   -- Init neural net
   math.randomseed(1337)
   net = nn.new_neural_net({
      neuron_counts = {1, 100, 1},
      act_fns = {nn.sigmoid},
      d_act_fns = {nn.d_sigmoid},
   })
   training_opts = {
      epochs = 1,
      learning_rate = 0.00005,
      log_freq = 0.5
   }

   -- Desired curve
   actual_curve = {
      color = {0, 1, 1, 0.9},
      points = generate_points(-20, 20, desired_fn),
   }

   -- Curve predicted by the neural network
   predicted_curve = {
      color = {1, 1, 0, 0.9},
      points = generate_points(-20, 20, function(x)
         return nn.feedforward(net, {inputs = {x}})[1]
      end)
   }

   -- Generate training data
   training_data = {}
   for i=1, 100 do
      local x = randf(-10, 10)
      table.insert(training_data, {inputs={x}, outputs={desired_fn(x)}})
   end

   -- Init a graph for our curves
   graph = graphlove.new({
      print_info = true,
      curves = {actual_curve, predicted_curve}
   })
   graphlove.update(graph)
end

function love.draw()
   graphlove.draw(graph, WIDTH, HEIGHT)
   lg.print(("learning_rate = %.16f"):format(training_opts.learning_rate),
            0, 58)
end

local function get_scale_delta(scale, dt, sign)
   local dscale = 40
   local small_dscale = 0.3
   if math.abs(scale) < 1 then
      return sign * dt * small_dscale
   else
      return sign * dt * dscale
   end
end

local function get_offset_delta(off, dt, sign)
   local doff = 300
   return sign * dt * doff
end

local is_training = false
local is_silent_training = false

function love.keypressed(key, scancode, isrepeat)
   if key == "g" then
      is_training = not is_training
   end
end

function love.update(dt)
   local recalc_points = false

   local scale_dx, scale_dy = 0, 0
   local off_dx, off_dy = 0, 0

   -- Force recalculation
   if love.keyboard.isDown("r") then
      recalc_points = true
   end

   -- Scale y
   if love.keyboard.isDown("k") then
      scale_dy = get_scale_delta(graph.y_scale, dt, 1)
   elseif love.keyboard.isDown("j") then
      scale_dy = get_scale_delta(graph.y_scale, dt, -1)
   end
   graph.y_scale = graph.y_scale + scale_dy

   -- Scale x
   if love.keyboard.isDown("l") then
      scale_dx = get_scale_delta(graph.x_scale, dt, 1)
   elseif love.keyboard.isDown("h") then
      scale_dx = get_scale_delta(graph.x_scale, dt, -1)
   end
   graph.x_scale = graph.x_scale + scale_dx

   -- Move x_off
   if love.keyboard.isDown("left") then
      off_dx = get_offset_delta(graph.x_off, dt, 1)
   elseif love.keyboard.isDown("right") then
      off_dx = get_offset_delta(graph.x_off, dt, -1)
   end
   graph.x_off = graph.x_off + off_dx

   -- Move y_off
   if love.keyboard.isDown("up") then
      off_dy = get_offset_delta(graph.y_off, dt, 1)
   elseif love.keyboard.isDown("down") then
      off_dy = get_offset_delta(graph.y_off, dt, -1)
   end
   graph.y_off = graph.y_off + off_dy

   -- Adjust learning rate
   if love.keyboard.isDown("=") then
      training_opts.learning_rate = training_opts.learning_rate * 1.1
   elseif love.keyboard.isDown("-") then
      training_opts.learning_rate = training_opts.learning_rate * 0.9
   end

   -- Train
   if love.keyboard.isDown("space") then
      nn.train(net, training_data, training_opts)
      recalc_points = true
   end

   if is_training then
      nn.train(net, training_data, training_opts)
      recalc_points = true
   end

   recalc_points = recalc_points or
                   off_dx ~= 0 or off_dy ~= 0 or
                   scale_dx ~= 0 or scale_dy ~= 0

   if recalc_points then
      predicted_curve.points = generate_points(-20, 20, function(x)
         return nn.feedforward(net, {inputs = {x}})[1]
      end)
      graphlove.update(graph)
   end
end
