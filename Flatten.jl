Flatten(input::GraphNode) = BroadcastedOperator(Flatten, input)
forward(::BroadcastedOperator{typeof(Flatten)}, input) = let
    return reshape(input, :)
end

backward(node::BroadcastedOperator{typeof(Flatten)}, input, gradient) = let
    tuple(reshape(gradient, size(input)))
end
