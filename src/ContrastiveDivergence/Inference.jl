using ..RBMBase


function inferRBM(weight::RBMWeight, k::Int, samples::Int, warmupSamples::Int)
    visible = sampleFromProbs(rand(1, length(weight.visible)))
    for _ in 1:warmupSamples
        hidden = visibleToHidden(weight, visible, sample)
        visible = hiddenToVisible(weight, hidden, sample)
    end
    
    output = []
    for i in 1:(samples * k)
        hidden = visibleToHidden(weight, visible, sample)
        visible = hiddenToVisible(weight, hidden, sample)
        if i % k == 0
            push!(output, visible)
        end
    end

    return output
end