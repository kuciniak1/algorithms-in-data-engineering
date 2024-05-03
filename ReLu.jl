ReLu(input::GraphNode) = BroadcastedOperator(ReLu, input)
forward(::BroadcastedOperator{typeof(ReLu)}, input) = let
    zeros_cons = zeros(size(input))
    output = max.(input, zeros_cons)
    return output
end

backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float64, 1}, g) = let
    J = input .> 0
    tuple(J .* g)
end


backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float64, 3}, g) = let
    J = input .> 0
    tuple(J .* g)
end
