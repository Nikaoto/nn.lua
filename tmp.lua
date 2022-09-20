-- A crude graphing calculator
local abs = math.abs
local pow = math.pow
local floor = math.floor
local f = function(x) return pow(x, 3)*2 + 18 end
local f2 = function(x) return 24*x - 14 end
local f3 = function(x) return x end
local y_scale = 10
local x_scale = 160
local points = {}
local cartesian_plain_ox = 300
local cartesian_plain_oy = 500
local recalc_points = false
local vline_girth = 1
local vline_length = 10
WIDTH = love.graphics.getWidth()
HEIGHT = love.graphics.getHeight()

function vert_line(x, y)
   local h = vline_length
   local w = vline_girth

   for x=x-floor(w/2), x+floor(w/2), 1 do
      love.graphics.line(
         x, y - math.floor(h/2),
         x, y + math.floor(h/2)
      )
   end
end

function horiz_line(x, y)
   local w = vline_length
   local h = vline_girth

   for y=y-floor(h/2), y+floor(h/2), 1 do
      love.graphics.line(
         x - math.floor(w/2), y,
         x + math.floor(w/2), y
      )
   end
end

function calc_points()
   -- f3(x) = f'(x)
   -- Calc f'(2)
   local x1 = 2
   local x2 = 2.001
   local y1 = f(x1)
   local y2 = f(x2)
   local slope = (y2-y1)/(x2-x1)
   f3 = function (x) return slope*x -14 end

   points = {}
   for x=-cartesian_plain_ox, love.graphics.getWidth(), 0.005 do
      local y = f(x)*y_scale
      if y > -cartesian_plain_oy and y < WIDTH then
         table.insert(points, {x=x*x_scale, y=y})
      end

      y = f2(x)*y_scale
      if y > -cartesian_plain_oy and y < WIDTH then
         table.insert(points, {x=x*x_scale, y=y})
      end

      y = f3(x)*y_scale
      if y > -cartesian_plain_oy and y < WIDTH then
         table.insert(points, {x=x*x_scale, y=y})
      end
   end
end

function love.conf(t)
   t.console = true
end

function love.load()
   calc_points()
   WIDTH = love.graphics.getWidth()
   HEIGHT = love.graphics.getHeight()
end

function love.draw()
   -- Draw y line
   love.graphics.setColor(1, 0, 0)
   love.graphics.line(
      cartesian_plain_ox, 0,
      cartesian_plain_ox, HEIGHT)
   -- Draw x line
   love.graphics.setColor(0, 0, 1)
   love.graphics.line(
          0, cartesian_plain_oy,
      WIDTH, cartesian_plain_oy)

   -- Draw vert lines on x line
   local xs
   if x_scale < 1 then
      love.graphics.setColor(1, 0.27, 0)
      xs = x_scale * 100
   else
      love.graphics.setColor(1, 0, 1)
      xs = x_scale
   end
   -- Draw positive side
   for x=cartesian_plain_ox, WIDTH, xs do
      vert_line(math.floor(x), cartesian_plain_oy)
   end
   -- Draw negative side
   for x=cartesian_plain_ox, 0, -xs do
      vert_line(math.floor(x), cartesian_plain_oy)
   end

   -- Draw horiz lines on y line
   local ys
   if y_scale < 1 then
      love.graphics.setColor(1, 0.27, 0)
      ys = y_scale * 100
   else
      love.graphics.setColor(1, 0, 1)
      ys = y_scale
   end
   if ys >= 1 then
      -- Draw positive side
      for y=cartesian_plain_oy, 0, -ys do
         horiz_line(cartesian_plain_ox, math.floor(y))
      end
      -- Draw negative side
      for y=cartesian_plain_oy, HEIGHT, ys do
         horiz_line(cartesian_plain_ox, math.floor(y))
      end
   end

   -- Draw all points of the graph (and lines in between)
   for i, p in ipairs(points) do
      love.graphics.setColor(1, 1, 1)
      love.graphics.points(
         cartesian_plain_ox + p.x,
         cartesian_plain_oy - p.y)

         -- love.graphics.setColor(0.3,0.3,0.3)
         -- love.graphics.line(
         --    cartesian_plain_ox + points[i].x,
         --    cartesian_plain_oy - points[i].y,
         --    cartesian_plain_ox + points[i+1].x,
         --    cartesian_plain_oy - points[i+1].y)
   end

   love.graphics.setColor(1, 0, 0)
   local x = cartesian_plain_ox + 2*x_scale
   local y = cartesian_plain_oy - f(2)*y_scale
   love.graphics.line(0, y, WIDTH, y)
   love.graphics.line(x, 0, x, HEIGHT)

   love.graphics.setColor(1, 1, 1)
   love.graphics.print(
      string.format(
         "x_scale = %.4f\ny_scale = %.4f",
         x_scale, y_scale))
end

function update_scale(scale, dt, sign)
   local dscale = 3
   local small_dscale = 0.3
   if abs(scale) < 1 then
      return scale + sign * dt * small_dscale
   else
      return scale + sign * dt * dscale
   end
end

function love.update(dt)
   if love.keyboard.isDown("up") then
      y_scale = update_scale(y_scale, dt, 1)
      recalc_points = true
   elseif love.keyboard.isDown("down") then
      y_scale = update_scale(y_scale, dt, -1)
      recalc_points = true
   end

   if love.keyboard.isDown("right") then
      x_scale = update_scale(x_scale, dt, 1)
      recalc_points = true
   elseif love.keyboard.isDown("left") then
      x_scale = update_scale(x_scale, dt, -1)
      recalc_points = true
   end

   if recalc_points then
      calc_points()
      recalc_points = false
   end
end
