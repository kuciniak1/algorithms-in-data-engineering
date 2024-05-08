ReLu(input::GraphNode) = BroadcastedOperator(ReLu, input)
forward(::BroadcastedOperator{typeof(ReLu)}, input) = let
    return max.(input, 0)
end

backward(node::BroadcastedOperator{typeof(ReLu)}, input, g) = let
    tuple( (input .>= 0) .* g)
end