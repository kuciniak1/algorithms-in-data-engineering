Flatten(input::GraphNode) = BroadcastedOperator(Flatten, input)
forward(::BroadcastedOperator{typeof(Flatten)}, input) = let
    return reshape(input, :)
end

backward(node::BroadcastedOperator{typeof(Flatten)}, input, g) = let
    J =  ones(Float32, size(input))
    gradient = reshape(g, size(input))
    J .*= gradient
    tuple(J)
end
