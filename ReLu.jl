mutable struct ReLuOperator{F} <: Operator
    inputs::Union{Nothing, Tuple{Operator}}
    output::Union{Nothing, Vector{Float32}, Array{Float32, 3}, Array{Float32, 1}}
    gradient::Union{Nothing, Array{Float32, 3}, Array{Float32, 1}}
    name::String
    ReLuOperator(fun, inputs...; name="?", output=output) = new{typeof(fun)}(inputs, output, nothing, name)
end


show(io::IO, x::ReLuOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");

ReLu(input::GraphNode) = let
    output = zeros(Float32, size(input.output))
    return ReLuOperator(ReLu, input; output=output)
end

forward(::ReLuOperator{typeof(ReLu)}, input) = let
    return max.(input, 0)
end

backward(node::ReLuOperator{typeof(ReLu)}, input::Array{Float32, 1}, g) = let
    tuple(g .* (input .>= 0))
end

backward(node::ReLuOperator{typeof(ReLu)}, input::Array{Float32, 3}, g) = let
    tuple(g .* (input .>= 0))
end
