using ..RBMBase


@enum OutputType logit prob sample

function sigmoid(x::Matrix{T} where T <: Real)
    return 1 ./ (1 .+ exp.(-x))
end

function sampleFromProbs(p::Matrix{T} where T <: Real; unsafe::Bool)
    if ~unsafe
        @assert maximum(p) <= 1.0
        @assert minimum(p) >= 0.0
    end
    return Int.(rand(size(p)...) .< p)
end

function sampleFromProbs(p::Matrix{T} where T <: Real)
    sampleFromProbs(p, unsafe=false)
end

function returnFromLogit(logit::Matrix{T} where T <: Real, outputType::OutputType, beta::Real)
    if outputType == logit
        return logit
    end

    p = sigmoid(beta .* logit)
    if outputType == prob
        return p
    end

    @assert outputType == sample
    return sampleFromProbs(p)
end


function visibleToHidden(weight::RBMWeight, visible::Matrix{T} where T <: Real, outputType::OutputType, beta::Real)
    logit = visible * weight.interaction
    return returnFromLogit(logit, outputType, beta)
end


function visibleToHidden(weight::RBMWeight, visible::Matrix{T} where T <: Real, outputType::OutputType)
    defaultBeta = 1.0
    return visibleToHidden(weight, visible, outputType, defaultBeta)
end


function hiddenToVisible(weight::RBMWeight, hidden::Matrix{T} where T <: Real, outputType::OutputType, beta::Real)
    logit = hidden * weight.interaction'
    return returnFromLogit(logit, outputType, beta)
end


function hiddenToVisible(weight::RBMWeight, hidden::Matrix{T} where T <: Real, outputType::OutputType)
    defaultBeta = 1.0
    hiddenToVisible(weight, hidden, outputType, defaultBeta)
end


function swapParallelChains!(weight::RBMWeight, betas::Vector{T} where T <: Real, visibles::Vector{Matrix{T}} where T <: Real, hiddens::Vector{Matrix{T}} where T <: Real)
    numChains = size(betas, 1)
    if numChains < 2
        return
    end
    sampledIdx = abs(rand(Int)) % (numChains - 1) + 1
    leftEnergy = rbmEnergy(weight, visibles[sampledIdx], hiddens[sampledIdx])
    rightEnergy = rbmEnergy(weight, visibles[sampledIdx + 1], hiddens[sampledIdx + 1])
    swapProb = exp.((betas[sampledIdx + 1] - betas[sampledIdx]) .* (rightEnergy - leftEnergy))
    swapIndices = sampleFromProbs(hcat(swapProb), unsafe=true)
    visibles[sampledIdx], visibles[sampledIdx + 1] = swapIndices .* visibles[sampledIdx + 1] + (1 .- swapIndices) .* visibles[sampledIdx], swapIndices .* visibles[sampledIdx] + (1 .- swapIndices) .* visibles[sampledIdx + 1]
    hiddens[sampledIdx], hiddens[sampledIdx + 1] = swapIndices .* hiddens[sampledIdx + 1] + (1 .- swapIndices) .* hiddens[sampledIdx], swapIndices .* hiddens[sampledIdx] + (1 .- swapIndices) .* hiddens[sampledIdx + 1]
end
