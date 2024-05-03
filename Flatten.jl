Flatten(input::GraphNode) = BroadcastedOperator(Flatten, input)
forward(::BroadcastedOperator{typeof(Flatten)}, input) = let
    return reshape(input, :)
end

backward(node::BroadcastedOperator{typeof(Flatten)}, input, g) = let
    height, width, channels = size(input)
    J =  ones(height, width, channels)
    g = reshape(g, height, width, channels)
    tuple(J .* g)
end
