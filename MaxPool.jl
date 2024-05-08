mutable struct MaxPoolOperator{F} <: Operator
    inputs::Union{Nothing, Tuple{GraphNode, GraphNode}}
    output::Union{Nothing, Array{Float32, 3}}
    gradient::Union{Nothing, Array{Float32, 3}}
    name::String
    MaxPoolOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end


show(io::IO, x::MaxPoolOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");

MaxPool(input::GraphNode, pool_size::GraphNode) = MaxPoolOperator(MaxPool, input, pool_size)

forward(::MaxPoolOperator{typeof(MaxPool)}, input, pool_size) = let
    input_rows, input_columns, channels = size(input)
    pool_height, pool_width = pool_size
    
    output_rows = div(input_rows, pool_height)
    output_columns = div(input_columns, pool_width)
    
    output = zeros(Float32, output_rows, output_columns, channels)
    
    for c in 1:channels
        for col in 1:output_columns
            for row in 1:output_rows
                row_start = 1 + (row-1)*pool_height
                row_end = row_start + pool_height-1
                col_start = 1 + (col-1)*pool_width
                col_end = col_start + pool_width-1
            
                pool = @view input[row_start:row_end, col_start:col_end, c]
                output[row, col, c] = maximum(pool)
            end
        end
    end
    return output
end


backward(node::MaxPoolOperator{typeof(MaxPool)}, input, pool_size, gradient) = let
    input_height, input_width, channels = size(input)
    pool_height, pool_width = pool_size
    gradient_height, gradient_width = size(gradient)
    
    input_height % pool_height != 0 ? input_height_new = pool_height*size(gradient)[1] : input_height_new = input_height
    input_width % pool_width != 0 ? input_width_new = pool_width*size(gradient)[2] : input_width_new = input_width
    
    J = zeros(Float32, input_height, input_width, channels)
        
    for c in 1:channels
        for i in 1:pool_width:input_width_new
            for j in 1:pool_height:input_height_new
                end_i = min(i + pool_width - 1, input_width)
                end_j = min(j + pool_height - 1, input_height)
            
                max_value, max_idx = findmax(@view input[i:end_i, j:end_j,c])

                J[i + max_idx[1] - 1, j + max_idx[2] - 1,c] = 1*gradient[div(i-1,pool_width) + 1, div(j-1, pool_height) + 1, c]
            end
        end
    end

    return tuple(J)
end
