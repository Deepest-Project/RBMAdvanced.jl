# RBMAdvanced

[![Build Status](https://github.com/Deepest-Project/RBMAdvanced.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Deepest-Project/RBMAdvanced.jl/actions/workflows/CI.yml?query=branch%3Amain)

RBMAdvanced.jl is a library for training RBMs with methods starting from naive training to advanced training methods such as parallel tempering.

## Usage Example

Below code trains a RBM on toy dataset and infers, using CD (contrastive divergence) method.

```
using RBMAdvanced

visibleSize = 2
hiddenSize = 3
dataset = [0 1; 0 0; 0 0]  # [0 1] for 1/3 probability, [0 0] otherwise

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
```

Check out our [documentation](docs/tutorial.md) to learn more about RBMs and how to train them.
