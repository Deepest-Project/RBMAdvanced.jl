Base.@kwdef struct TrainingConfig
    usePersistentChain::Bool
    betas::Vector{T} where T <: Real
    k::Int
    epochs::Int
    learningRate::Real
end


function validateConfig(config::TrainingConfig)
    if config.usePersistentChain
        if size(config.betas, 1) > 1
            println("Using PT")
        else
            println("Using PCD")
        end
    else
        @assert size(config.betas, 1) == 1 "PCD is required for training with PT"
        println("Using CD")
    end
end
