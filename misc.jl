function cross_entropy_loss(prediction, value)
    return sum(-value .* log.(prediction))
end

function glorot_uniform(size::Tuple{Int, Int, Int, Int}; scale=1.0f0)
    fan_in = size[1] * size[2] * size[3]
    fan_out = size[1] * size[2] * size[4]
    limit = sqrt(6 / (fan_in + fan_out)) * scale
    return randn(Float32, size) * Float32(limit)
end

function glorot_uniform(size::Tuple{Int, Int}; scale=1.0f0)
    fan_in = size[1]
    fan_out = size[2]
    limit = sqrt(6 / (fan_in + fan_out)) * scale
    return randn(Float32, size) * Float32(limit)
end

function onehot(target::Int64, labels::UnitRange{Int64})
    num_labels = length(labels)
    onehot = zeros(Float32, num_labels)
    if target in labels
        onehot[target - first(labels) + 1] = 1
    end
    return onehot
end

function onehotbatch(targets::Vector{Int64}, labels::UnitRange{Int64})
    num_targets = length(targets)
    num_labels = length(labels)
    onehot_matrix = zeros(Int, num_labels, num_targets)
    for i in 1:num_targets
        onehot_matrix[targets[i]+1, i] = 1
    end
    return onehot_matrix
end

function onecold(vector, labels)
    idx = argmax(vector)
    idx <= length(labels) ? labels[idx] : nothing
end
