using RBMAdvanced
using Test


visibleSize = 2
hiddenSize = 3
dataset = [0 1; 1 0; 0 1; 1 0; 0 1; 1 0]

@testset "Naive" begin
    weight = randInitRBMWeight(visibleSize, hiddenSize)
    trainRBM_naive!(weight, dataset, 100, 0.1)
    output = inferRBM_naive(weight)
end

@testset "CD" begin
    weight = randInitRBMWeight(visibleSize, hiddenSize)
    cdConfig = TrainingConfig(
        usePersistentChain=false,
        betas=[1.0],
        k=1,
        epochs=100,
        learningRate=0.1
    )
    validateConfig(cdConfig)
    trainRBM!(weight, dataset, cdConfig)
    output = RBMAdvanced.inferRBM(weight, 1, 100, 100)
end

@testset "PCD" begin
    weight = randInitRBMWeight(visibleSize, hiddenSize)
    pcdConfig = TrainingConfig(
        usePersistentChain=true,
        betas=[1.0],
        k=1,
        epochs=100,
        learningRate=0.1
    )
    validateConfig(pcdConfig)
    trainRBM!(weight, dataset, pcdConfig)
    output = RBMAdvanced.inferRBM(weight, 1, 100, 100)
end

@testset "PT" begin
    weight = randInitRBMWeight(visibleSize, hiddenSize)
    ptConfig = TrainingConfig(
        usePersistentChain=true,
        betas=[1.0, 0.1, 0.01],
        k=1,
        epochs=100,
        learningRate=0.1
    )
    validateConfig(ptConfig)
    trainRBM!(weight, dataset, ptConfig)
    output = RBMAdvanced.inferRBM(weight, 1, 100, 100)
end
