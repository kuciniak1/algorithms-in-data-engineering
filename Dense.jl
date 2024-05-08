mutable struct DenseOperator{F} <: Operator
    inputs::Union{Nothing, Tuple{Variable, Variable, Operator}, Tuple{Variable, Operator},}
    output::Union{Nothing, Array{Float32, 3}, Array{Float32, 1}, Array{Float32, 2}}
    gradient::Union{Nothing, Array{Float32, 3}, Array{Float32, 1}, Array{Float32, 2}}
    name::String
    DenseOperator(fun, inputs...; name="?", output=output) = new{typeof(fun)}(inputs, output, nothing, name)
end

Dense(weights::GraphNode, bias::GraphNode, input::GraphNode) = let
    height = size(bias.output)
    output = zeros(Float32, height)
    return DenseOperator(Dense, weights, bias, input; output=output)
end

Dense(weights::GraphNode, input::GraphNode) = let
    height = size(weights.output)[1]
    output = zeros(Float32, height)
    return DenseOperator(Dense, weights, input; output=output)
end

forward(::DenseOperator{typeof(Dense)}, weights, bias, input) = let
    return weights * input .+ bias
end

forward(::DenseOperator{typeof(Dense)}, weights, input) = let
    return weights * input
end

backward(node::DenseOperator{typeof(Dense)}, weights, bias, input, gradient) = let
    grad_weights = gradient * input' 
    grad_bias = sum(gradient, dims=1)
    grad_input = weights' * gradient
    
    return tuple(grad_weights, grad_bias, grad_input)
end

backward(node::DenseOperator{typeof(Dense)}, weights, input, gradient) = let
    grad_weights = input' * gradient
    grad_input = gradient * weights'
    return tuple(grad_weights, grad_bias, grad_input)
end
