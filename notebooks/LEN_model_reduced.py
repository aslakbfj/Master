

class LEN(nn.Module):
    def __init__(self, num_cnns, input_length, num_classes, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                    weight_path=None):
        
        self.linears = nn.Sequential(
            nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                        groups=num_cnns),
            nn.BatchNorm1d(num_cnns),
            ExpActivation(),
            nn.MaxPool1d(input_length - (filter_size-1)),
            
            #Replace Flatten with LEN
            
            nn.Flatten()
            )

        self.final = nn.Linear(num_cnns, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))


    def forward(self, x):
        x = x.repeat(1, self._options["num_cnns"], 1)
        outs = self.linears(x)
        out = self.final(outs)
        return out
