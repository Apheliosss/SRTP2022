
import math
import torch.nn as nn
class SWNN(nn.Module):
    def __init__(self, outsize):
        super(GNNWR, self).__init__()
        self.outsize = outsize
        thissize = 0
        lastsize = 512
        i = 2

        self.fc = nn.Sequential()
        self.fc.add_module("full"+str(1), nn.Linear(self.outsize, 600))


        while math.pow(2, int(math.log2(lastsize))) >= max(128, outsize + 1):
            if i == 1:
                thissize = int(math.pow(2, int(math.log2(lastsize))))
            else:
                thissize = int(math.pow(2, int(math.log2(lastsize)) - 1))
            
            self.fc.add_module("full"+str(i), nn.Linear(lastsize, thissize))
            self.fc.add_module("batc"+str(i), nn.BatchNorm1d(thissize))
            self.fc.add_module("acti"+str(i), nn.PReLU(init=0.4))
            self.fc.add_module("drop"+str(i), nn.Dropout(0.2))

            lastsize = thissize
            i = i + 1

        self.fc.add_module("full"+str(i), nn.Linear(lastsize, outsize))
        
    def forward(self, x):
        x = self.fc(x)
        return x