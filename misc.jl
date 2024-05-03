function cross_entropy_loss(prediction, value)
    return sum(-value .* log.(prediction))
end

function glorot_uniform(size::Tuple{Int, Int, Int, Int}; scale=1.0)
    fan_in = size[1] * size[2] * size[3]
    fan_out = size[1] * size[2] * size[4]
    limit = sqrt(6 / (fan_in + fan_out)) * scale
    return randn(size) * limit
end

function glorot_uniform(size::Tuple{Int, Int}; scale=1.0)
    fan_in = size[1]
    fan_out = size[2]
    limit = sqrt(6 / (fan_in + fan_out)) * scale
    return randn(size) * limit
end

function onehot(target::Int64, labels::UnitRange{Int64})
    num_labels = length(labels)
    onehot = zeros(num_labels)
    idx = findfirst(isequal(target), labels)
    onehot[idx] = 1
    return onehot
end

function onecold(vector, labels)
    argmax(vector) <= length(labels) ? labels[argmax(vector)] : nothing
end
