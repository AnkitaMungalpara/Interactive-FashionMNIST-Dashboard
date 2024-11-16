import torch.nn as nn

class FashionMNISTNet(nn.Module):
    def __init__(self, layer_sizes=[512, 256, 128]):
        super(FashionMNISTNet, self).__init__()
        
        layers = [nn.Flatten()]
        input_size = 28 * 28
        
        for size in layer_sizes:
            layers.extend([
                nn.Linear(input_size, size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = size
            
        layers.append(nn.Linear(input_size, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 