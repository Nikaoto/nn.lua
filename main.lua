-- Graph a function and find its local minimum using a simple
-- naive dy/dx calculation
local inspect = require("inspect")
local nn = require("nn")
require("util")

local lg = love.graphics
local abs = math.abs
local pow = math.pow
local sqrt = math.sqrt
local floor = math.floor
local f = function(x) return x*x end
local y_scale = 80
local x_scale = 40
local x_off, y_off = -30, -140
local points = {}
local tangent_points = {}
local cartesian_plain_ox = 300
local cartesian_plain_oy = 500
local recalc_points = false
local vline_girth = 1
local vline_length = 10
local px = 6 -- will slowly descend
local step = 1
local tangent_slope = 0
WIDTH = lg.getWidth()
HEIGHT = lg.getHeight()
local canvas = lg.newCanvas(WIDTH, HEIGHT)
local recording = false
local frame_idx = 1

function vert_line(x, y)
   local h = vline_length
   local w = vline_girth

   for x=x-floor(w/2), x+floor(w/2), 1 do
      lg.line(
         x, y - math.floor(h/2),
         x, y + math.floor(h/2)
      )
   end
end

function horiz_line(x, y)
   local w = vline_length
   local h = vline_girth

   for y=y-floor(h/2), y+floor(h/2), 1 do
      lg.line(
         x - math.floor(w/2), y,
         x + math.floor(w/2), y
      )
   end
end

function calc_points()
   points = {}
   for x=-cartesian_plain_ox, lg.getWidth()/3, 0.01 do
      local y = f(x)*y_scale
      if y > -cartesian_plain_oy and y < WIDTH then
         table.insert(points, {x=x*x_scale, y=y})
      end
   end

   predicted_points = {}
   if f2 then
      for x=-cartesian_plain_ox, lg.getWidth()/3, 0.01 do
         y = f2(x)*y_scale
         if y > -cartesian_plain_oy and y < WIDTH then
            table.insert(predicted_points, {x=x*x_scale, y=y})
         end
      end
   end
end

local function generate_training_data(fn, n, seed)
   local data = {}
   math.randomseed(seed or os.time())
   for i=1, n do
      local a = randf(interval_min, interval_max)
      local ans = fn(a)
      data[i] = { inputs={a}, outputs={ans}}
   end
   return data
end

function love.conf(t)
   t.console = true
end

function love.load()
   calc_points()
   WIDTH = lg.getWidth()
   HEIGHT = lg.getHeight()

   net = nn.new_neural_net({
      seed = os.time(),
      neuron_counts = {1, 8, 1 },
      weight_min = -10, weight_max = 10,
      act_fns = {nil, relu, nil },
      d_act_fns = {nil, d_relu, nil },
   })

   training_opts = {
      epochs = 10,
      learning_rate = 0.0000001,
      log_freq = 0.5
   }

   interval_min = -3
   interval_max = 3

   training_data = generate_training_data(f, 100)

   f2 = function(x)
      return nn.feedforward(net, {inputs={x}})[1]
   end
end

function love.draw()
   if recording then
      lg.setCanvas(canvas)
      lg.clear(0, 0, 0, 1)
   end
   -- Draw y line
   lg.setColor(1, 0, 0)
   lg.line(
      cartesian_plain_ox + x_off, 0,
      cartesian_plain_ox + x_off, HEIGHT)
   -- Draw x line
   lg.setColor(0, 0, 1)
   lg.line(
          0, cartesian_plain_oy + y_off,
      WIDTH, cartesian_plain_oy + y_off)

   -- Draw vert lines on x line
   local xs
   if x_scale < 1 then
      lg.setColor(1, 0.27, 0)
      xs = x_scale * 100
   else
      lg.setColor(1, 0, 1)
      xs = x_scale
   end
   -- Draw positive side
   for x=cartesian_plain_ox+x_off, WIDTH, xs do
      vert_line(math.floor(x), cartesian_plain_oy+y_off)
   end
   -- Draw negative side
   for x=cartesian_plain_ox+x_off, 0, -xs do
      vert_line(math.floor(x), cartesian_plain_oy+y_off)
   end

   -- Draw horiz lines on y line
   local ys
   if y_scale < 1 then
      lg.setColor(1, 0.27, 0)
      ys = y_scale * 100
   else
      lg.setColor(1, 0, 1)
      ys = y_scale
   end
   if ys >= 1 then
      -- Draw positive side
      for y=cartesian_plain_oy+y_off, 0, -ys do
         horiz_line(cartesian_plain_ox+x_off, math.floor(y))
      end
      -- Draw negative side
      for y=cartesian_plain_oy+y_off, HEIGHT, ys do
         horiz_line(cartesian_plain_ox+x_off, math.floor(y))
      end
   end

   -- Draw all points of the graph
   lg.setColor(1, 1, 1, 0.5)
   for i, p in ipairs(points) do
      lg.points(
         cartesian_plain_ox + p.x + x_off,
         cartesian_plain_oy - p.y + y_off)
   end

   -- Draw all points of the predicted fn graph
   lg.setColor(0, 1, 1, 0.9)
   for i, p in ipairs(predicted_points) do
      lg.circle(
         "fill",
         cartesian_plain_ox + p.x + x_off,
         cartesian_plain_oy - p.y + y_off, 1)
   end

   -- Print stats
   lg.setColor(1, 1, 1)
   lg.print(
      string.format(
         "x_scale = %.4f\ny_scale = %.4f\nx_off = %.4f\ny_off = %.4f\nlearning_rate = %.16f",
         x_scale, y_scale, x_off, y_off, training_opts.learning_rate))
   if recording then
      lg.setCanvas()
      canvas:newImageData():encode("png", string.format("%03i.png", frame_idx))
      frame_idx = frame_idx + 1
   end
   love.graphics.draw(canvas)
end

function update_scale(scale, dt, sign)
   local dscale = 40
   local small_dscale = 0.3
   if abs(scale) < 1 then
      return scale + sign * dt * small_dscale
   else
      return scale + sign * dt * dscale
   end
end

function update_offset(off, dt, sign)
   local doff = 100
   return off + sign * dt * doff
end

function love.update(dt)
   if love.keyboard.isDown("k") then
      y_scale = update_scale(y_scale, dt, 1)
      recalc_points = true
   elseif love.keyboard.isDown("j") then
      y_scale = update_scale(y_scale, dt, -1)
      recalc_points = true
   end

   if love.keyboard.isDown("l") then
      x_scale = update_scale(x_scale, dt, 1)
      recalc_points = true
   elseif love.keyboard.isDown("h") then
      x_scale = update_scale(x_scale, dt, -1)
      recalc_points = true
   end

   if love.keyboard.isDown("r") then
      recalc_points = true
   end

   if love.keyboard.isDown("left") then
      x_off = update_offset(x_off, dt, -1)
      recalc_points = true
   elseif love.keyboard.isDown("right") then
      x_off = update_offset(x_off, dt, 1)
      recalc_points = true
   end

   if love.keyboard.isDown("up") then
      y_off = update_offset(y_off, dt, -1)
      recalc_points = true
   elseif love.keyboard.isDown("down") then
      y_off = update_offset(y_off, dt, 1)
      recalc_points = true
   end

   if love.keyboard.isDown("space") then
      nn.train(net, training_data, training_opts)
      recalc_points = true
   end

   if love.keyboard.isDown("g") then
      is_training = true
   end

   if love.keyboard.isDown("s") then
      is_training = false
      is_silent_training = false
   end

   if love.keyboard.isDown("w") then
      is_silent_training = true
   end

   if love.keyboard.isDown("=") then
      training_opts.learning_rate = training_opts.learning_rate / (10*dt)
   end

   if love.keyboard.isDown("-") then
      training_opts.learning_rate = training_opts.learning_rate * (10*dt)
   end

   if is_training then
      nn.train(net, training_data, training_opts)
      recalc_points = true
   end
   if is_silent_training then
      nn.train(net, training_data, training_opts)
   end

   if recalc_points then
      calc_points()
      recalc_points = false
   end
end
