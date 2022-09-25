local graphlove = require("graphlove")

local WIDTH, HEIGHT = 1024, 720
local lg = love.graphics
local function sq(x) return x*x end

local graph

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

   training_opts = {
      epochs = 10,
      learning_rate = 0.0000001,
      log_freq = 0.5
   }

   graph = graphlove.new({
      print_info = true,
      curves = {{
         color = {0, 1, 1, 0.9},
         points = generate_points(-100, 100, math.sin),
      }, {
         color = {1, 1, 0, 0.9},
         points = generate_points(-100, 100, sq)
      }
   }})

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
   local doff = 100
   return sign * dt * doff
end


function love.update(dt)
   local should_update_graph = false

   local scale_dx, scale_dy = 0, 0
   local off_dx, off_dy = 0, 0

   -- Force update graph
   if love.keyboard.isDown("r") then
      should_update_graph = true
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
      training_opts.learning_rate = training_opts.learning_rate / (10*dt)
   elseif love.keyboard.isDown("-") then
      training_opts.learning_rate = training_opts.learning_rate * (10*dt)
   end

   -- if is_training then
   --    nn.train(net, training_data, training_opts)
   --    recalc_points = true
   -- end
   -- if is_silent_training then
   --    nn.train(net, training_data, training_opts)
   -- end


   should_update_graph = should_update_graph or
                         off_dx ~= 0 or off_dy ~= 0 or
                         scale_dx ~= 0 or scale_dy ~= 0
   if should_update_graph then
      graphlove.update(graph)
   end
end
