local inspect = require("inspect")
local nn = require("nn")

-- Stub for running with lua only
if not love then love = { graphics = print } end

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

local lg = love.graphics
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

-- training_data = {
--  { inputs = {image data}, outputs = {label number} },
--  { inputs = {image data}, outputs = {label number} },
--  ...
-- }
local training_data = {}
-- Maps label number to an array of neuron values for the output layer
local label_map = {
   [0] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
   [1] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
   [2] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
   [3] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
   [4] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
   [5] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
   [6] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
   [7] = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
   [8] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
   [9] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
}
local img
local label

function copy_arr(arr, from, to)
   local new_arr = {}
   for i=from, to do
      table.insert(new_arr, arr[i])
   end
   return new_arr
end

function load_training_data()
   local count = 10
   -- Read labels
   local lbls = eat(labels_file, count)
   local imgs = eat(images_file, COLS*ROWS * count)

   for i=1, count do
      local image = copy_arr(imgs, (i-1)*ROWS*COLS, i*ROWS*COLS)
      table.insert(training_data, { inputs = image, outputs = label_map[lbls[i]] })
   end
end

function next_img()
   image_ind = image_ind + 1

   img = training_data[image_ind].inputs
   label = training_data[image_ind].outputs[1]

   -- -- Read label
   -- local lbl = eat(labels_file, 1)
   -- label = lbl[1] or -1

   -- -- Read image
   -- img = eat(images_file, COLS*ROWS)

   -- return img
end

--[[
Step 1: Load all images into memory
Step 2: Create neural net
Step 3: Train neural net using loaded images & labels
Step 4: Save neural net
Step 5: Test neural net against training and testing set
]]--

load_training_data()
math.randomseed(os.time())
local net = nn.new_neural_net({
   neuron_counts = {ROWS*COLS, 128, 10},
   act_fns = {"sigmoid", "sigmoid"}
})

nn.train(net, training_data, {
   epochs = 100,
   learning_rate = 1,
   annealing = 0.99,
   log_freq = 1,
})

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
   for i, pix in ipairs(img) do
      lg.setColor(1, 1, 1, pix/255)
      local col = i % COLS
      local row = (i - col) / ROWS
      lg.rectangle(
         "fill",
         image_x + col*SZ,
         image_y + row*SZ,
         SZ, SZ)
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
