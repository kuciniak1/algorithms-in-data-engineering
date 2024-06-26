mutable struct ConvOperator{F} <: Operator
    inputs::Union{Nothing, Tuple{GraphNode, GraphNode, GraphNode}, Tuple{GraphNode, GraphNode}}
    output::Union{Nothing, Array{Float32, 3}}
    gradient::Union{Nothing, Array{Float32, 3}}
    name::String
    input_2d::Union{Nothing, Array{Float32, 2}}
    weights_2d::Union{Nothing, Array{Float32, 2}}
    output_2d::Union{Nothing, Array{Float32, 2}}
    ConvOperator(fun, inputs...; output=nothing, name="?", input_2d=nothing, weights_2d=nothing, output_2d=nothing) = new{typeof(fun)}(inputs, output, nothing, name, input_2d, weights_2d, output_2d)
end


show(io::IO, x::ConvOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");

Conv(input::GraphNode, weights::GraphNode, bias::GraphNode) = let
    input_height, input_width, input_channels = size(input.output)
    kernel_height, kernel_width, channels_in, channels_out = size(weights.output)
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    output = zeros(Float32, output_height, output_width, channels_out)
    
    input_2d = zeros(Float32, input_height, input_width)
    weights_2d = zeros(Float32, kernel_height, kernel_width)
    output_2d = zeros(Float32, output_height, output_width)
    return ConvOperator(Conv, input, weights, bias; output=output, input_2d=input_2d, weights_2d=weights_2d, output_2d=output_2d)
end


forward(node::ConvOperator{typeof(Conv)}, input, weights, bias) = let
    kernel_height, kernel_width, input_channels, output_channels = size(weights)
    
    sumret = zeros(Float32, kernel_height, kernel_width)
    
    for k in 1:output_channels
        for c in 1:input_channels
            node.input_2d .= @views input[:, :, c]
            node.weights_2d .= @views weights[:, :, c, k]
            Convolution_2d!(node.output_2d, sumret, node.input_2d, node.weights_2d; bias=bias[k])
            @views node.output[:, :, k] .= node.output_2d
            node.output_2d .= 0.0
        end
    end
    return @views node.output
end

backward(node::ConvOperator{typeof(Conv)}, input, weights, bias, gradient) = let
    input_height, input_width, input_channels = size(input)
    output_height, output_width, output_channels = size(gradient)
    kernel_height, kernel_width, _, _ = size(weights)
    
    grad_input = zeros(Float32, size(input))
    grad_weights = zeros(Float32, size(weights))
    
    for k in 1:input_channels
        for c in 1:output_channels
            for i = 1:output_height
                for j = 1:output_width
                    @views grad_input[i:i+kernel_height-1, j:j+kernel_width-1, k] .+= (weights[:, :, k, c] .* gradient[i, j, c]);
                end
            end
        end
    end
    
    sumret = zeros(Float32, output_height, output_width)
    for k in 1:input_channels
        for c in 1:output_channels
            node.weights_2d .= 0
            node.input_2d .= @views input[:, :, k]
            node.output_2d .= @views gradient[:, :, c]
            Convolution_2d!(node.weights_2d, sumret, node.input_2d, node.output_2d)
            @views grad_weights[:, :, k, c] .+= node.weights_2d
        end
    end

    grad_bias = reshape(sum(gradient, dims=(1,2,4)), :)
    
    return tuple(grad_input, grad_weights, grad_bias)
end


forward(node::ConvOperator{typeof(Conv)}, input, weights) = let
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, _, output_channels = size(weights)

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output = zeros(Float32, output_height, output_width, output_channels)
    ret = zeros(Float32, output_height, output_width)
    sumret = zeros(Float32, kernel_height, kernel_width)
    tmp_input = zeros(Float32, input_height, input_width)
    tmp_weights = zeros(Float32, kernel_height, kernel_width)
    
    for k in 1:output_channels
        for c in 1:input_channels
            tmp_input .= @views input[:, :, c]
            tmp_weights .= @views weights[:, :, c, k]
            Convolution_2d!(ret, sumret, tmp_input, tmp_weights)
            @views output[:, :, k] .+= ret
            ret .= 0.0
        end
    end
    return output
end

backward(node::ConvOperator{typeof(Conv)}, input, weights, gradient) = let
    input_height, input_width, input_channels = size(input)
    output_height, output_width, output_channels = size(gradient)
    kernel_height, kernel_width, _, _ = size(weights)
    
    grad_input = zeros(Float32, size(input))
    grad_weights = zeros(Float32, size(weights))
    
    tmp_weights = zeros(Float32, kernel_height, kernel_width)
    tmp_input = zeros(Float32, input_height, input_width)
    tmp_gradient = zeros(Float32, output_height, output_width)
    for k in 1:input_channels
        for c in 1:output_channels
            for i = 1:output_height
                for j = 1:output_width
                    @views grad_input[i:i+kernel_height-1, j:j+kernel_width-1, k] .+= (weights[:, :, k, c] .* gradient[i, j, c]);
                end
            end
        end
    end
    sumret = zeros(Float32, output_height, output_width)
    for k in 1:input_channels
        for c in 1:output_channels
            tmp_weights .= 0
            tmp_input .= @views input[:, :, k]
            tmp_gradient .= @views gradient[:, :, c]
            Convolution_2d!(tmp_weights, sumret, tmp_input, tmp_gradient)
            @views grad_weights[:, :, k, c] .+= tmp_weights
        end
    end
    
    return tuple(grad_input, grad_weights)
end



function Convolution_2d(input, kernel; bias=0., padding=false)
    input_rows, input_columns = size(input)
    kernel_height, kernel_width = size(kernel)

    if padding
        padded_input = zeros(Float32, input_rows + 2*kernel_height - 2, input_columns + 2*kernel_width - 2)
        padded_input[kernel_height:end-kernel_height+1, kernel_width:end-kernel_width+1] .= input
        input_rows, input_columns = size(padded_input)
        input = padded_input
    end

    output_rows = input_rows - kernel_height + 1
    output_columns = input_columns - kernel_width + 1
    output = zeros(Float32, output_rows, output_columns)
    sumret = zeros(size(kernel))
    for c in 1:output_columns
        for r in 1:output_rows
            patch = @view input[r:r+kernel_height-1, c:c+kernel_width-1]
            sumret .= patch .* kernel
            output[r, c] = sum(sumret) + bias
            sumret .= 0.0
        end
    end
    return output
end

function Convolution_2d!(ret, sumret, input, kernel; bias=0., padding=false)
    input_rows, input_columns = size(input)
    kernel_height, kernel_width = size(kernel)

    if padding
        padded_input = zeros(Float32, input_rows + 2*kernel_height - 2, input_columns + 2*kernel_width - 2)
        padded_input[kernel_height:end-kernel_height+1, kernel_width:end-kernel_width+1] .= input
        input_rows, input_columns = size(padded_input)
        input = padded_input
    end

    output_rows = input_rows - kernel_height + 1
    output_columns = input_columns - kernel_width + 1
    for c in 1:output_columns
        for r in 1:output_rows
            patch = @view input[r:r+kernel_height-1, c:c+kernel_width-1]
            @views sumret .= patch .* kernel
            ret[r, c] = sum(sumret) + bias
            sumret .= 0.0
        end
    end
end
