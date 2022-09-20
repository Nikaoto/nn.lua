-- util

function shuffle(arr)
   for i=#arr, 1, -1 do
      local j = math.random(1, i)
      arr[j], arr[i] = arr[i], arr[j]
   end
end

function sq(n) return n*n end

function lerp(a, b, p)
   return a + (b - a) * p
end

function randf(min, max)
   return lerp(min, max, math.random(0, 10000) / 10000)
end

function map(arr, fn)
   local new_arr = {}
   for i, _ in ipairs(arr) do
      new_arr[i] = fn(arr[i], i)
   end
   return new_arr
end

function rand_arr(min, max, n, seed)
   math.randomseed(seed or os.time())
   local arr = {}
   for i=1, n do
      table.insert(
         arr,
         lerp(min, max, math.random(0, 10000) / 10000))
   end
   return arr
end

function linear(w)
   return w
end

function relu(a)
   return a > 0 and a or 0
end

function sigmoid(a)
   return 1 / (1 + math.exp(-a))
end

function tanh(a)
   return math.tanh(a)
end

function d_relu(a)
   return a > 0 and 1 or 0
end

function d_sigmoid(a)
   local s = sigmoid(a)
   return s * (1 - s)
end
