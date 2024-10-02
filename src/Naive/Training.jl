using LinearAlgebra

using ..RBMBase

function trainRBM_naive!(weight::RBMWeight, dataset::Matrix{T} where T <: Real, epochs::Integer, lr::Real)
    for epoch in 1:epochs
        v = size(weight.visible, 1)
        h = size(weight.hidden, 1)

        z_q = zeros(2^v)
        for visibleIndex in 1:2^v
            for hiddenIndex in 1:2^h
                state = numberToRBMState(weight, visibleIndex, hiddenIndex)
                z_q[visibleIndex] += exp(-rbmEnergy(weight, state))
            end
        end
        z = sum(z_q)

        P = zeros(2^v)
        datasetSize = size(dataset, 1)
        for i in 1:datasetSize
            P[binaryStateToIndex(dataset[i, :])] += 1 / datasetSize
        end

        PP = zeros(2^v, 2^h)
        QQ = zeros(2^v, 2^h)
        for visibleIndex in 1:2^v
            for hiddenIndex in 1:2^h
                state = numberToRBMState(weight, visibleIndex, hiddenIndex)
                probfactor = exp(-rbmEnergy(weight, state))
                PP[visibleIndex, hiddenIndex] = P[visibleIndex] * probfactor / z_q[visibleIndex]
                QQ[visibleIndex, hiddenIndex] = probfactor / z
            end
        end
        
        d_visible = zeros(v)
        d_hidden = zeros(h)
        d_interaction = zeros(v, h)
        for visibleIndex in 1:2^v
            for hiddenIndex in 1:2^h
                state = numberToRBMState(weight, visibleIndex, hiddenIndex)
                diff = PP[visibleIndex, hiddenIndex] - QQ[visibleIndex, hiddenIndex]
                d_visible += state.visible .* diff
                d_hidden += state.hidden .* diff
                d_interaction += state.visible * state.hidden' .* diff
            end
        end
        weight.visible += lr .* d_visible
        weight.hidden += lr .* d_hidden
        weight.interaction += lr .* d_interaction
    end
end
