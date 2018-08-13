import random
import torch
from torch.autograd import Variable
 
x_tdata = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_tdata = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
 
 
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
 
model = LinearRegressionModel()
 
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
 
for epoch in range(5001):
    pred_y = model(x_tdata)
 
    loss = criterion(pred_y, y_tdata)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))

new_var = random.randrange(1.0, 99.0)

pred_x = Variable(torch.Tensor([[new_var]]))
pred_y = model(pred_x)
print("predict ", new_var, model(pred_x).data[0][0])
