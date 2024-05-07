using Random

mutable struct DataLoader{T}
    data::T
    batch_size::Int
    shuffle::Bool
    index::Vector{Int}
end

function DataLoader(data, batch_size::Int, shuffle= false)
    index = collect(1:size(data[1], 4))
    if shuffle
        Random.shuffle!(index)
    end
    return DataLoader(data, batch_size, shuffle, index)
end


function Base.iterate(dataloader::DataLoader)
    if isempty(dataloader.index)
        return nothing
    end
    
    @views batch_size = min(length(dataloader.index), dataloader.batch_size)
    @views batch_indices = dataloader.index[1:batch_size]
    @views dataloader.index = dataloader.index[batch_size+1:end]  # Update index for next iteration
    
    batch_features = []
    batch_labels = []
    for idx in batch_indices
        @views features = dataloader.data[1][:, :, :, idx]
        @views labels = dataloader.data[2][:, idx]
        push!(batch_features, features)
        push!(batch_labels, labels)
    end
    return (batch_features, batch_labels), 1
    
end


function Base.iterate(dataloader::DataLoader, state)
    if isempty(dataloader.index)
        return nothing
    end
    
    @views batch_size = min(length(dataloader.index), dataloader.batch_size)
    @views batch_indices = dataloader.index[1:batch_size]
    @views dataloader.index = dataloader.index[batch_size+1:end]  # Update index for next iteration
    
    batch_features = []
    batch_labels = []
    for idx in batch_indices
        @views features = dataloader.data[1][:, :, :, idx]
        @views labels = dataloader.data[2][:, idx]
        push!(batch_features, features)
        push!(batch_labels, labels)
    end
    return (batch_features, batch_labels), 1
end