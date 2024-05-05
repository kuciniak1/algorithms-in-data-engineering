function Conv(input, weights, bias, activation) return activation.(Convolution(input, weights, bias)) end

function Conv(input, weights, activation) return activation.(Convolution(input, weights)) end


Convolution(input::GraphNode, weights::GraphNode, bias::GraphNode) = BroadcastedOperator(Convolution, input, weights, bias)
forward(::BroadcastedOperator{typeof(Convolution)}, input, weights, bias) = let
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, _, output_channels = size(weights)

    output = zeros(Float32, input_height - kernel_height + 1, input_width - kernel_width + 1, output_channels)
    tmp_input = zeros(Float32, input_height, input_width)
    tmp_weights = zeros(Float32, kernel_height, kernel_width)
    
    for k in 1:output_channels
        for c in 1:input_channels
            tmp_input .= @views input[:, :, c]
            tmp_weights .= @views weights[:, :, c, k]
            output[:, :, k] .+= Convolution_2d(tmp_input, tmp_weights; bias=bias[k])
        end
    end
    return output
end

backward(node::BroadcastedOperator{typeof(Convolution)}, input, weights, bias, gradient) = let
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
            tmp_weights .= @views weights[:, :, k, c]
            tmp_gradient .= @views gradient[:, :, c]
            for i = 1:output_height
                for j = 1:output_width
                    grad_input[i:i+kernel_height-1, j:j+kernel_width-1, k] .+= (weights[:,:,k,c] .* gradient[i,j,c]);
                end
            end
        end
    end
    
    for k in 1:input_channels
        for c in 1:output_channels
            tmp_input .= @views input[:, :, k]
            tmp_gradient .= @views gradient[:, :, c]
            grad_weights[:, :, k, c] += Convolution_2d(tmp_input, tmp_gradient)
        end
    end

    grad_bias = reshape(sum(gradient, dims=(1,2,4)), :)
    
    return grad_input, grad_weights, grad_bias
end



Convolution(input::GraphNode, weights::GraphNode) = BroadcastedOperator(Convolution, input, weights)
forward(::BroadcastedOperator{typeof(Convolution)}, input, weights) = let
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, _, output_channels = size(weights)

    output = zeros(Float32, input_height - kernel_height + 1, input_width - kernel_width + 1, output_channels)
    tmp_input = zeros(Float32, input_height, input_width)
    tmp_weights = zeros(Float32, kernel_height, kernel_width)
    
    for k in 1:output_channels
        for c in 1:input_channels
            tmp_input .= @views input[:, :, c]
            tmp_weights .= @views weights[:, :, c, k]
            output[:, :, k] .+= Convolution_2d(tmp_input, tmp_weights)
        end
    end
    return output
end

backward(node::BroadcastedOperator{typeof(Convolution)}, input, weights, gradient) = let
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
            tmp_weights .= @views weights[:, :, k, c]
            tmp_gradient .= @views gradient[:, :, c]
            grad_input[:, :, k] += Convolution_2d(tmp_weights, tmp_gradient; padding=true)
        end
    end

    for k in 1:input_channels
        for c in 1:output_channels
            tmp_input .= @views input[:, :, k]
            tmp_gradient .= @views gradient[:, :, c]
            grad_weights[:, :, k, c] += Convolution_2d(tmp_input, tmp_gradient)
        end
    end
    
    return grad_input, grad_weights
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

    for c in 1:output_columns
        for r in 1:output_rows
            patch = @view input[r:r+kernel_height-1, c:c+kernel_width-1]
            output[r, c] = sum(patch .* kernel) + bias
        end
    end
    return output
end
