mutable struct FlattenOperator{F} <: Operator
    inputs::Union{Nothing, Tuple{Operator}}
    output::Union{Nothing, Array{Float32, 1}}
    gradient::Union{Nothing, Array{Float32, 1}}
    name::String
    FlattenOperator(fun, inputs...; name="?", output=output) = new{typeof(fun)}(inputs, output, nothing, name)
end


Flatten(input::GraphNode) = let
    output = reshape(input.output, :)
    return FlattenOperator(Flatten, input; output=output)
end

forward(::FlattenOperator{typeof(Flatten)}, input) = let
    return reshape(input, :)
end

backward(node::FlattenOperator{typeof(Flatten)}, input, gradient) = let
    tuple(reshape(gradient, size(input)))
end
