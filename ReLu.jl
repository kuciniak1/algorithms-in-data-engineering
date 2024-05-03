ReLu(input::GraphNode) = BroadcastedOperator(ReLu, input)
forward(::BroadcastedOperator{typeof(ReLu)}, input) = let
    zeros_cons = zeros(size(input))
    output = max.(input, zeros_cons)
    return output
end

backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float64, 1}, g) = let
    height = length(input)
    J = zeros(height)
    for i in 1:height
        if input[i] > 0
            J[i] = 1
        end
    end
    tuple(J.* g)
end


backward(node::BroadcastedOperator{typeof(ReLu)}, input::Array{Float64,3}, g) = let
    height, width, channels = size(input)
    J = zeros(height, width, channels)
    for i in 1:height
        for j in 1:width
            for c in 1:channels
                if input[i,j,c] > 0
                    J[i,j,c] = 1
                end
            end
        end
    end
    tuple(J.* g)
end
