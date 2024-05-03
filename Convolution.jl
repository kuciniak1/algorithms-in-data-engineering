function Conv(input, weights, bias, activation) return activation.(Convolution(input, weights, bias)) end

function Conv(input, weights, activation) return activation.(Convolution(input, weights)) end


Convolution(input::GraphNode, weights::GraphNode, bias::GraphNode) = BroadcastedOperator(Convolution, input, weights, bias)
forward(::BroadcastedOperator{typeof(Convolution)}, input, weights, bias) = let
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, _, output_channels = size(weights)

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1    
    output = zeros(output_height, output_width, output_channels)

    for k in 1:output_channels
        for c in 1:input_channels
            output[:, :, k] += Convolution_2d(input[:, :, c], weights[:, :, c, k], bias[k])
        end
    end
    
    return output
end

backward(node::BroadcastedOperator{typeof(Convolution)}, input, weights, bias, gradient) = let
    input_height, input_width, input_channels = size(input) # 13x13x6
    output_height, output_width, output_channels = size(gradient) # 11x11x16
    kernel_height, kernel_width, _, _ = size(weights) # 3x3x6x16
    
    
    grad_input = zeros(size(input)) # 13x13x6
    for k in 1:input_channels # 6
        for c in 1:output_channels # 16
            grad_input[:, :, k] += Convolution_2d(weights[:, :, k, c], gradient[:, :, c], 0; padding=true)
        end
    end

    
    grad_kernel = zeros(size(weights)) # 3x3x1x6
    for k in 1:input_channels #1
        for c in 1:output_channels #6
            grad_kernel[:, :, k, c] += Convolution_2d(input[:, :, k], gradient[:, :, c], 0)
        end
    end

    grad_bias = reshape(sum(gradient, dims=(1,2,4)), :)
    
    return grad_input, grad_kernel, grad_bias
end



Convolution(input::GraphNode, weights::GraphNode) = BroadcastedOperator(Convolution, input, weights)
forward(::BroadcastedOperator{typeof(Convolution)}, input, weights) = let
    input_height, input_width, input_channels = size(input)
    kernel_height, kernel_width, _, output_channels = size(weights)

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1    
    output = zeros(output_height, output_width, output_channels)

    for k in 1:output_channels
        for c in 1:input_channels
            output[:, :, k] += Convolution_2d(input[:, :, c], weights[:, :, c, k])
        end
    end
    
    return output
end

backward(node::BroadcastedOperator{typeof(Convolution)}, input, weights, gradient) = let
    input_height, input_width, input_channels = size(input) # 13x13x6
    output_height, output_width, output_channels = size(gradient) # 11x11x16
    kernel_height, kernel_width, _, _ = size(weights) # 3x3x6x16
    
    
    grad_input = zeros(size(input)) # 13x13x6
    for k in 1:input_channels # 6
        for c in 1:output_channels # 16
            grad_input[:, :, k] += Convolution_2d(weights[:, :, k, c], gradient[:, :, c]; padding=true)
        end
    end

    grad_kernel = zeros(size(weights)) # 3x3x1x6
    for k in 1:input_channels #1
        for c in 1:output_channels #6
            grad_kernel[:, :, k, c] += Convolution_2d(input[:, :, k],gradient[:, :, c])
        end
    end

    return grad_input, grad_kernel
end



function Convolution_2d(input, kernel, bias; padding=false)
    input_height, input_width = size(input)
    kernel_height, kernel_width = size(kernel)

    if padding
        tmp = zeros(input_height+2*kernel_height-2, input_width+2*kernel_width-2)
        for i in 1:input_height
            for j in 1:input_width
                tmp[i+kernel_height-1, j+kernel_width-1] = input[i,j]
            end
        end
        input = tmp
        input_height, input_width = size(input)
    end


    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1    
    output = zeros(output_height, output_width)

    for i in 1:output_height
        for j in 1:output_width
            patch = input[i:i+kernel_height-1, j:j+kernel_width-1]
            output[i, j] += sum(patch .* kernel) .+ bias
        end
    end
    return output
end


function Convolution_2d(input, kernel; padding=false)
    input_height, input_width = size(input)
    kernel_height, kernel_width = size(kernel)

    if padding
        tmp = zeros(input_height+2*kernel_height-2, input_width+2*kernel_width-2)
        for i in 1:input_height
            for j in 1:input_width
                tmp[i+kernel_height-1, j+kernel_width-1] = input[i,j]
            end
        end
        input = tmp
        input_height, input_width = size(input)
    end


    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1    
    output = zeros(output_height, output_width)

    for i in 1:output_height
        for j in 1:output_width
            patch = input[i:i+kernel_height-1, j:j+kernel_width-1]
            output[i, j] += sum(patch .* kernel)
        end
    end
    return output
end

