import torch
import random
from torch import nn
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from openpyxl import Workbook

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the data

torch.manual_seed(42)


start = 0
stop = 10
step = 0.1



X = torch.arange(start, stop, step)
y = torch.sin(X)



X_train, X_test, y_train, y_test = train_test_split(X.unsqueeze(dim=1),
                                                    y.unsqueeze(dim=1),
                                                    test_size=0.2,
                                                    random_state=42)



class LineBreakerV0(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=1,
                      out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5,
                      out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5,
                      out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5,
                      out_features=1)
        )
        
    def forward(self, x):
        return self.layer_stack(x)
    
model_1 = LineBreakerV0()

loss_fn = nn.MSELoss()
optimizer1 = torch.optim.SGD(params=model_1.parameters(),
                            lr = 0.01)

epochs = 10000

epoch_num = []
train_losses = []
test_losses = []

def Train_Model(model,
                epochs,
                loss_fn,
                optimizer):

  for epoch in range(epochs):
      
      model.train()
      
      # Forward pass
      y_pred = model(X_train)
      
      # Calculate loss
      loss = loss_fn(y_pred, y_train)
      
      # Optim zero grad
      optimizer.zero_grad()
      
      # Loss backward
      loss.backward()
      
      # Optimizer step
      optimizer.step()
      
      # Testing
      model.eval()
      with torch.inference_mode():
          
          test_pred = model(X_test)
          
          test_loss = loss_fn(test_pred, y_test)
          
          if epoch % (epochs/10) == 0:
              
              print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}")
              epoch_num.append(epoch)
              train_losses.append(loss.numpy())
              test_losses.append(test_loss.numpy())

Train_Model(model=model_1,
            epochs=epochs,
            loss_fn=loss_fn,
            optimizer=optimizer1)
