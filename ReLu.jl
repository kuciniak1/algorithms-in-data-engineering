ReLu(input::GraphNode) = BroadcastedOperator(ReLu, input)
forward(::BroadcastedOperator{typeof(ReLu)}, input) = let
    zeros_cons = zeros(Float32, size(input))
    output = max.(input, zeros_cons)
    return output
end

backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float32, 1}, g) = let
    J = input .> 0
    tuple(J .* g)
end


backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float32, 3}, g) = let
    J = input .> 0
    tuple(J .* g)
end
