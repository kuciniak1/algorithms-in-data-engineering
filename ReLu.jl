ReLu(input::GraphNode) = BroadcastedOperator(ReLu, input)
forward(::BroadcastedOperator{typeof(ReLu)}, input) = let
    zeros_cons = zeros(Float32, size(input))
    output = max.(input, zeros_cons)
    return output
end

backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float32, 1}, g) = let
    tuple(g .* (input .>= 0))
end


backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float32, 3}, g) = let
    tuple(g .* (input .>= 0))
end
