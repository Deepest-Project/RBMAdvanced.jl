using LinearAlgebra
using Statistics

using ..RBMBase

function trainRBM!(weight::RBMWeight, dataset::Matrix{T} where T <: Real, config::TrainingConfig)
    v = size(weight.visible, 1)
    h = size(weight.hidden, 1)
    N = size(dataset, 1)

    if config.usePersistentChain
        persistentVisibleByBetas = [sampleFromProbs(randLike(dataset)) for _ in config.betas]
        persistentHiddenByBetas = [visibleToHidden(weight, v, sample) for v in persistentVisibleByBetas]
    else
        persistentHiddenByBetas = []
    end

    for epoch in 1:config.epochs
        d_visible = zerosLike(weight.visible)
        d_hidden = zerosLike(weight.hidden)
        d_interaction = zerosLike(weight.interaction)

        # coldest
        h0 = visibleToHidden(weight, dataset, sample)

        if config.usePersistentChain
            negInit = persistentHiddenByBetas[1]
        else
            negInit = h0
        end

        v1 = hiddenToVisible(weight, negInit, sample)
        h1 = visibleToHidden(weight, v1, sample)

        if config.usePersistentChain
            persistentVisibleByBetas[1] = v1
            persistentHiddenByBetas[1] = h1

            for chainIdx in 2:size(config.betas, 1)
                beta = config.betas[chainIdx]
                persistentVisibleByBetas[chainIdx] = hiddenToVisible(weight, persistentHiddenByBetas[chainIdx], sample, beta)
                persistentHiddenByBetas[chainIdx] = visibleToHidden(weight, persistentVisibleByBetas[chainIdx], sample, beta)
            end
    
            swapParallelChains!(weight, config.betas, persistentVisibleByBetas, persistentHiddenByBetas)

            v1 = persistentVisibleByBetas[1]
            h1 = persistentHiddenByBetas[1]
        end

        d_visible = d_visible + (mean(dataset, dims=1) - mean(v1, dims=1))[1, :]
        d_hidden = d_hidden + (mean(h0, dims=1) - mean(h1, dims=1))[1, :]
        d_interaction = d_interaction + (dataset' * h0 - v1' * h1) ./ N
        
        weight.visible += d_visible .* config.learningRate
        weight.hidden += d_hidden .* config.learningRate
        weight.interaction += d_interaction .* config.learningRate
    end
end
