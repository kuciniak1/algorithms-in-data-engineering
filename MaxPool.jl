MaxPool(input::GraphNode, pool_size) = BroadcastedOperator(MaxPool, input, pool_size)
forward(::BroadcastedOperator{typeof(MaxPool)}, input, pool_size) = let
    input_height, input_width, channels = size(input)
    pool_height, pool_width = pool_size
    
    output_height = div(input_height, pool_height)
    output_width = div(input_width, pool_width)
    
    output = zeros(output_height, output_width, channels)
    
    for c in 1:channels
        for i in 1:output_height
            for j in 1:output_width
                row_start = 1 + (i-1)*pool_height
                row_end = row_start + pool_height-1
                col_start = 1 + (j-1)*pool_width
                col_end = col_start + pool_width-1
            
                pool = input[row_start:row_end, col_start:col_end, c]
                output[i, j, c] = maximum(pool)
            end
        end
    end
    return output
end


backward(node::BroadcastedOperator{typeof(MaxPool)}, input, pool_size, gradient) = let
    input_height, input_width, channels = size(input) # 11x11x16
    pool_height, pool_width = pool_size # 2x2
    gradient_height, gradient_width = size(gradient) # 5x5x16
    
    input_height % pool_height != 0 ? input_height_new = pool_height*size(gradient)[1] : input_height_new = input_height
    input_width % pool_width != 0 ? input_width_new = pool_width*size(gradient)[2] : input_width_new = input_width
    
    J = zeros(input_height, input_width, channels) # 11x11x16
        
    for c in 1:channels
        for i in 1:pool_width:input_width_new # 1, 3, 5, 7, 9, 11
            for j in 1:pool_height:input_height_new # 1, 3, 5, 7, 9, 11
                end_i = min(i + pool_width - 1, input_width) # 2
                end_j = min(j + pool_height - 1, input_height) # 2
            
                max_value, max_idx = findmax(input[i:end_i, j:end_j,c]) # [1:2, 1:2, 1, 1]

                J[i + max_idx[1] - 1, j + max_idx[2] - 1,c] = 1*gradient[div(i-1,pool_width) + 1, div(j-1, pool_height) + 1, c]
            end
        end
    end

    return tuple(J)
end
