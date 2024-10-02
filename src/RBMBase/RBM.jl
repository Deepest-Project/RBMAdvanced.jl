mutable struct RBMWeight
    visible::Array
    hidden::Array
    interaction::Matrix
end

struct RBMState
    visible::Array
    hidden::Array
end

function randInitRBMWeight(visibleSize::Integer, hiddenSize::Integer)
    visible = Array(rand(visibleSize))
    hidden = Array(rand(hiddenSize))
    interaction = rand(visibleSize, hiddenSize)
    return RBMWeight(visible, hidden, interaction)
end
