using ..RBMBase

function inferRBM_naive(weight::RBMWeight)
    v = size(weight.visible, 1)
    h = size(weight.hidden, 1)

    z = 0.
    for visibleIndex in 1:2^v
        for hiddenIndex in 1:2^h
            state = numberToRBMState(weight, visibleIndex, hiddenIndex)
            z += exp(-rbmEnergy(weight, state))
        end
    end

    Q_unnormalized = zeros(2^v)
    for visibleIndex in 1:2^v
        for hiddenIndex in 1:2^h
            state = numberToRBMState(weight, visibleIndex, hiddenIndex)
            Q_unnormalized[visibleIndex] += exp(-rbmEnergy(weight, state))
        end
    end
    Q = Q_unnormalized / z

    Q_dict = Dict()
    for visibleIndex in 1:2^v
        Q_dict[indexToBinaryState(v, visibleIndex)] = Q[visibleIndex]
    end
    
    return Q_dict
end
