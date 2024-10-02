using LinearAlgebra

function rbmEnergy(weight::RBMWeight, state::RBMState)
    visibledot = weight.visible ⋅ state.visible
    hiddendot = weight.hidden ⋅ state.hidden
    interaction = (state.visible' * weight.interaction * state.hidden)[1]
    return -(visibledot + hiddendot + interaction)
end

function rbmEnergy(weight::RBMWeight, visible::Matrix, hidden::Matrix)
    visibledot = visible * weight.visible
    hiddendot = hidden * weight.hidden
    interaction = sum((visible * weight.interaction) .* hidden, dims=2)[:, 1]
    return -(visibledot + hiddendot + interaction)
end

function indexToBinaryState(size::Integer, index::Integer)
    return digits(index - 1, base=2, pad=size)
end

function binaryStateToIndex(state::Vector{Int})
    return foldl((x, y) -> 2x + y, reverse(state)) + 1
end

function numberToRBMState(weight::RBMWeight, visibleIndex::Integer, hiddenIndex::Integer)
    v = size(weight.visible, 1)
    h = size(weight.hidden, 1)
    visible = indexToBinaryState(v, visibleIndex)
    hidden = indexToBinaryState(h, hiddenIndex)
    return RBMState(visible, hidden)
end


function zerosLike(x::Matrix)
    return zeros(size(x)...)
end


function zerosLike(x::Array)
    return Array(zeros(size(x)))
end


function randLike(x::Matrix)
    return rand(size(x)...)
end
