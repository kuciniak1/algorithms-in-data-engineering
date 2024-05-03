function Dense(w, b, x, activation) return activation(w * x .+ b) end
function Dense(w, x, activation) return activation(w * x) end
