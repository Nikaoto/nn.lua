local inspect = require("inspect")

--[[
Image file format

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel


]]--

local SZ = 10
local ROWS, COLS = 28, 28

local file = io.open("train-images-idx3-ubyte", "r")
file:seek("set", 14)

function eat(file, nbytes)
   local str = file:read(nbytes)
   return {str:byte(1, #str)}
end

local img

function next_img()
   img = {}
   for i=1, ROWS do
      table.insert(img, eat(file, COLS))
   end
end

function love.load()
   next_img()
end

function love.draw()
   for col, _ in ipairs(img) do
      for row, _ in ipairs(img[col]) do
         love.graphics.setColor(1, 1, 1, img[row][col]/255)
         love.graphics.rectangle(
            "fill",
            col*SZ, row*SZ,
            SZ, SZ)
      end
   end
end

function love.keypressed(key)
   if key == "space" then
      next_img()
   end
end
