local inspect = require("inspect")
local lg = love.graphics

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

local IMAGE_COUNT = 60000
local SZ = 10
local ROWS, COLS = 28, 28

local IMAGES_FILE_LOC_START = 16
local LABELS_FILE_LOC_START = 8
local images_file_loc = IMAGES_FILE_LOC_START
local labels_file_loc = LABELS_FILE_LOC_START
local label_x, label_y = 30, 30
local ind_x, ind_y = label_x, label_y + 20
local image_x, image_y = label_x, ind_y + 50
local image_ind = 0

local images_file = io.open("train-images-idx3-ubyte", "r")
images_file:seek("set", images_file_loc)

local labels_file = io.open("train-labels-idx1-ubyte", "r")
labels_file:seek("set", labels_file_loc)

function eat(file, nbytes)
   local str = file:read(nbytes)
   return {str:byte(1, #str)}
end

local img
local label

function next_img()
   image_ind = image_ind + 1

   -- Read label
   local lbl = eat(labels_file, 1)
   label = lbl[1] or -1

   -- Read image
   img = {}
   for i=1, ROWS do
      table.insert(img, eat(images_file, COLS))
   end
end

function love.load()
   next_img()
end

function love.draw()
   lg.setColor(1, 1, 1, 1)
   -- Draw index
   lg.print(("image_ind = %d/%d"):format(image_ind, IMAGE_COUNT), ind_x, ind_y, 0, 2, 2)
   -- Draw label
   lg.print(("label = %d"):format(label), label_x, label_y, 0, 2, 2)
   
   -- Draw image outline
   lg.setColor(0, 1, 1, 1)
   lg.rectangle("line", image_x, image_y, SZ*COLS, SZ*ROWS)

   -- Draw image
   for col, _ in ipairs(img) do
      for row, _ in ipairs(img[col]) do
         lg.setColor(1, 1, 1, img[row][col]/255)
         lg.rectangle(
            "fill",
            image_x + col*SZ,
            image_y + row*SZ,
            SZ, SZ)
      end
   end
end

function love.keypressed(key)
   if key == "space" then
      next_img()
   elseif key == "backspace" then
      image_ind = image_ind - 2
      labels_file_loc = labels_file:seek("cur") - 2
      images_file_loc = images_file:seek("cur") - COLS * ROWS * 2

      if labels_file_loc < LABELS_FILE_LOC_START then
         labels_file_loc = LABELS_FILE_LOC_START
      end
      if images_file_loc < IMAGES_FILE_LOC_START then
         images_file_loc = IMAGES_FILE_LOC_START
      end
      if image_ind < 0 then image_ind = 0 end

      labels_file:seek("set", labels_file_loc)
      images_file:seek("set", images_file_loc)
      next_img()
   end
end
