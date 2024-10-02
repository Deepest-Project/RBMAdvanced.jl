module RBMAdvanced

module RBMBase
    include("RBMBase/RBM.jl")
    include("RBMBase/Utils.jl")
    
    export
        RBMWeight,
        RBMState,
        randInitRBMWeight,
        indexToBinaryState,
        binaryStateToIndex,
        numberToRBMState,
        rbmEnergy,
        zerosLike,
        randLike
end

module ContrastiveDivergence
    include("ContrastiveDivergence/Config.jl")
    include("ContrastiveDivergence/Training.jl")
    include("ContrastiveDivergence/Inference.jl")
    include("ContrastiveDivergence/Utils.jl")

    export
        TrainingConfig,
        TrainingMethod,
        trainRBM!,
        inferRBM,
        validateConfig,
        swapParallelChains!
end

module Naive
    include("Naive/Training.jl")
    include("Naive/Inference.jl")

    export
        trainRBM_naive!,
        inferRBM_naive
end

using .RBMBase
using .Naive
using .ContrastiveDivergence

export
    randInitRBMWeight,
    TrainingConfig,
    TrainingMethod,
    trainRBM_naive!,
    inferRBM_naive,
    trainRBM!,
    inferRBM,
    validateConfig

end
