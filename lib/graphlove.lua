local graphlove = {}

local gl = graphlove
local lg = love.graphics
local abs = math.abs
local pow = math.pow
local sqrt = math.sqrt
local floor = math.floor

local function draw_vert_crossing(x, y, crossing_length, crossing_girth)
   local h = crossing_length
   local w = crossing_girth

   for x=x-floor(w/2), x+floor(w/2), 1 do
      lg.line(
         x, y - floor(h/2),
         x, y + floor(h/2)
      )
   end
end

local function draw_horiz_crossing(x, y, crossing_length, crossing_girth)
   local w = crossing_length
   local h = crossing_girth

   for y=y-floor(h/2), y+floor(h/2), 1 do
      lg.line(
         x - floor(w/2), y,
         x + floor(w/2), y
      )
   end
end

--[[
   Curve opts table spec: {
      color = {R, G, B, A},
      points = {x1, y1, x2, y2, x3...},

      -- Optional
      point_radius = 2,
   }
]]-- 
function gl.new(opts)
   local graph = {
      print_info = opts.print_info or false,
      plane_ox = opts.plane_ox or 300,
      plane_oy = opts.plane_oy or 500,
      x_off = opts.x_off or 0,
      y_off = opts.y_off or 0,
      y_scale = opts.y_scale or 80,
      x_scale = opts.x_scale or 40,
      crossing_girth = opts.crossing_girth or 1,
      crossing_length = opts.crossing_length or 10,
      x_axis_color = opts.x_axis_color or {0, 0, 1, 1},
      y_axis_color = opts.y_axis_color or {1, 0, 0, 1},
      crossing_color = opts.crossing_color or {1, 0, 1, 0.9},
      alt_crossing_color = opts.alt_crossing_color or {1, 0.27, 0, 0.9},
      SCALE_COLOR_CHANGE_THRESHOLD = 1, -- FIXME: rename this
      curves = opts.curves,
   }

   return graph
end

-- Remap all points according to the scale and offset. Should be called after
-- any changes have been made to the curves passed to the graph.
function gl.update(graph)
   for _, curve in ipairs(graph.curves) do
      curve.raw_points = {}
      for i=1, #curve.points, 2 do
         local x = graph.plane_ox + graph.x_off +
                   curve.points[i] * graph.x_scale
         local y = graph.plane_oy + graph.y_off -
                   curve.points[i+1] * graph.y_scale
         table.insert(curve.raw_points, x)
         table.insert(curve.raw_points, y)
      end
   end
end

-- TODO: add and use graph_x and graph_y
function gl.draw(graph, width, height)
   -- Draw y axis line
   lg.setColor(graph.y_axis_color)
   lg.line(
      graph.plane_ox + graph.x_off, 0,
      graph.plane_ox + graph.x_off, height)

   -- Draw x axis line
   lg.setColor(graph.x_axis_color)
   lg.line(
          0, graph.plane_oy + graph.y_off,
      width, graph.plane_oy + graph.y_off)

   -- Draw vertical crossing lines (unit markers) on x axis
   local xs
   if graph.x_scale < graph.SCALE_COLOR_CHANGE_THRESHOLD then
      lg.setColor(graph.alt_crossing_color)
      xs = graph.x_scale * 100 -- FIXME: annoying hardcode
   else
      lg.setColor(graph.crossing_color)
      xs = graph.x_scale
   end
   -- Draw positive side
   for x=graph.plane_ox + graph.x_off, width, xs do
      draw_vert_crossing(
         floor(x),
         graph.plane_oy + graph.y_off,
         graph.crossing_length,
         graph.crossing_girth)
   end
   -- Draw negative side
   for x=graph.plane_ox + graph.x_off, 0, -xs do
      draw_vert_crossing(
         floor(x),
         graph.plane_oy + graph.y_off,
         graph.crossing_length,
         graph.crossing_girth)
   end

   -- Draw horiz crossing lines (unit markers) on y axis
   local ys
   if graph.y_scale < graph.SCALE_COLOR_CHANGE_THRESHOLD then
      lg.setColor(graph.alt_crossing_color)
      ys = graph.y_scale * 100 -- FIXME: annoying hardcode
   else
      lg.setColor(graph.crossing_color)
      ys = graph.y_scale
   end
   if ys >= 1 then
      -- Draw positive side
      for y=graph.plane_oy + graph.y_off, 0, -ys do
         draw_horiz_crossing(
            graph.plane_ox + graph.x_off,
            floor(y),
            graph.crossing_length,
            graph.crossing_girth)
      end
      -- Draw negative side
      for y=graph.plane_oy + graph.y_off, height, ys do
         draw_horiz_crossing(
            graph.plane_ox + graph.x_off,
            floor(y),
            graph.crossing_length,
            graph.crossing_girth)
      end
   end

   -- Draw all curves
   for _, curve in ipairs(graph.curves) do
      lg.setColor(curve.color)

      -- Draw circles if point radius given
      if curve.radius and curve.radius > 1 then
         for i=1, #curve.raw_points, 2 do
            lg.circle(
               "fill",
               curve.raw_points[i],
               curve.raw_points[i+1],
               curve.radius)
         end
      else  -- Draw points otherwise
         lg.points(curve.raw_points)
      end
   end

   -- Print stats
   if graph.print_info then
      lg.setColor(1, 1, 1, 1)
      lg.print(string.format(
         "x_scale = %.4f\ny_scale = %.4f\nx_off = %.4f\ny_off = %.4f",
         graph.x_scale, graph.y_scale, graph.x_off, graph.y_off))
   end
end

return graphlove
