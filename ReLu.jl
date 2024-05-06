ReLu(input::GraphNode) = BroadcastedOperator(ReLu, input)
forward(::BroadcastedOperator{typeof(ReLu)}, input) = let
    return max.(input, 0)
end

backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float32, 1}, g) = let
    tuple(g .* (input .>= 0))
end


backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float32, 3}, g) = let
    tuple(g .* (input .>= 0))
end
