import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import torch
from torch import nn
import torch.optim as optim
import os
import time
start_time = time.time()
random_seed = 10
# Set random seeds for reproducibility
np.random.seed(random_seed)  # Set seed for NumPy
torch.manual_seed(random_seed)  # Set seed for PyTorch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # Set seed for all GPUs (if applicable)

# Generate MG time series
def generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions):
    samplelength = sequenceLength
    initWashoutLength = sequenceLength // 10

    inputSequence = np.ones((sequenceLength, dimensions))
    outputSequence = np.ones((sequenceLength, dimensions))

    incrementsperUnit = 10
    genHistoryLength = tau * incrementsperUnit
    seed = 1.2 * np.ones((int(genHistoryLength), dimensions)) + 0.2 * (np.random.rand(int(genHistoryLength), dimensions) - 0.5)
    oldval = 1.2 * np.ones((1, dimensions))
    genHistory = seed
    speedup = 1

    sample = np.zeros((samplelength, dimensions))
    step = 0

    for n in range(1, samplelength + initWashoutLength + 1):
        for i in range(incrementsperUnit * speedup):
            step += 1
            tauval = genHistory[int(step % genHistoryLength)]
            newval = oldval + (0.2 * tauval / (1.0 + tauval ** 10) - 0.1 * oldval) / incrementsperUnit
            genHistory[int(step % genHistoryLength)] = oldval
            oldval = newval

        if n > initWashoutLength:
            sample[n - initWashoutLength - 1] = newval

    testseq = sample - 1.0
    outputSequence = testseq
    inputSequence[1:] = outputSequence[:-1]
    inputSequence[0] = 0.0

    return inputSequence, outputSequence

# Example usage:
sequenceLength = 20000
tau = 17
dimensions = 1
inputSequence, outputSequence = generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions)
print("Input Sequence Shape:", inputSequence.shape)
print("Output Sequence Shape:", outputSequence.shape)

nrmse_value = []
fitness_value = []
counter = 0


# code of model


class ESN(nn.Module):
    def __init__(self, input_size, res_size, output_size, a, reg, W):
        super(ESN, self).__init__()
        self.input_size = input_size
        self.res_size = res_size
        self.output_size = output_size
        self.a = a
        self.reg = reg
        self.W = W
        np.random.seed(42)
        numpy_array1 = np.random.rand(res_size, 1 + input_size)
        # print(f'Win : {numpy_array1}')
        self.Win = nn.Parameter((torch.tensor(numpy_array1) - 0.5) * 1)
        # W = W.reshape((res_size, res_size))
        rhoW = max(abs(linalg.eig(W.detach().numpy())[0]))
        self.W.data *= 1.25 / rhoW



    def forward(self, data, trainLen, testLen, initLen, errorLen):
      torch.set_printoptions(precision=12)
      X = torch.zeros((1 + self.input_size + self.res_size, trainLen - initLen))
      Yt = data[None, initLen + 1:trainLen + 1]
      Yt = torch.tensor(Yt)
      x = torch.zeros((self.res_size, 1))
      for t in range(trainLen):
          u = data[t]
          # print(f'u : {u}')
          scalar = torch.tensor([1])
          input_1U = torch.cat((scalar,u)).unsqueeze(1)
          x = (1 - self.a) * x + self.a *  torch.tanh(torch.mm(self.Win, input_1U) + torch.mm(self.W, x))
          # print(f'x : {x}')
          if t >= initLen:
              scalar =torch.tensor([1])
              out = torch.cat((scalar, u , x.squeeze()))
              # print(f'out : {out}')
              X[:, t - initLen] = out.squeeze()
              # print(f'X : {X[2][0]}')

      A = torch.mm(X, X.T) + self.reg * torch.eye(self.input_size + self.res_size + 1)
      # print(f'X : {X.shape}')
      # print(f'y : {Yt.shape}')
      yt_t = torch.squeeze(Yt, dim=0)
      B = torch.mm(X, yt_t)
      # print(f'yt_t : {yt_t}')

      Wout = torch.linalg.solve(A, B).T
  # --------------------------------------------------------
      Y = torch.zeros((self.output_size, testLen))
      u = data[trainLen]
      # print(f'x type 4 {W.dtype}')
      # print(f'requires_grad 4  {W.requires_grad}')
      for t in range(testLen):
          scalar = torch.tensor([1])
          input_1U = torch.cat((scalar,u)).unsqueeze(1)
          x = (1 - self.a) * x + self.a *  torch.tanh(torch.mm(self.Win, input_1U) + torch.mm(self.W, x))
          scalar =torch.tensor([1])
          out = torch.cat((scalar, u , x.squeeze()))
          # print(f'Wout : {Wout.shape}')
          # print(f'out : {out.shape}')
          y = torch.mv(Wout, out)
          Y[:, t] = y
          u = y


      return Y.T

def claculation_nrmse(predict_tensor, actual_tensor):
    mse = torch.mean((predict_tensor - actual_tensor) ** 2)
    rmse = torch.sqrt(mse)
    print(f'mse : {mse}')
    range = torch.max(actual_tensor) - torch.min(actual_tensor)
    nrmse = rmse / range
    print(f'nrmse ------>  ; {nrmse}')
    return nrmse


def loss_func(x):
    # data1 = np.loadtxt('MackeyGlass_t17.txt')
    data = outputSequence[:14000]
    input_size = output_size = 1
    res_size = 1000
    initLen = 100
    errorLen = 500
    num_reservoirs = 1
    a = 0.3
    reg = 4.840e-07
    pre_train = 0.2
    pre_test = 0.8
    len_of_data = len(data)
    trainLen = int(pre_train * len_of_data)
    testLen = len_of_data - trainLen

    torch.set_default_dtype(torch.double)

    np.random.seed(42)
    numpy_array1 = np.random.rand(res_size, 1 + input_size)

    Win = nn.Parameter((torch.tensor(numpy_array1) - 0.5) * 1)
    W = x
    W = W.reshape((res_size, res_size))

    data_test = data[trainLen:]
    data_train = data[:trainLen]

    torch.set_default_dtype(torch.double)
    data = torch.tensor(data)

    esn = ESN(input_size, res_size, output_size, a, reg, W)
    prediction = esn(data, trainLen, testLen, initLen, errorLen)
    # print(f'predictions : {prediction}')

    # print(f'  prediction ===================== {prediction}')
    data_tensor = torch.tensor(data)
    pred  =  prediction
    actual =  data_tensor[trainLen + 1: trainLen + errorLen + 1, :]
    plt.figure(figsize=(12, 6))

    # plt.plot(pred.detach().numpy(), label='Predicted train',color='red')
    # plt.plot(actual.detach().numpy(), label='Actual train',color = 'blue')
    # plt.title('train')
    # plt.show()
    # nrmse = claculation_nrmse(prediction, data_tensor[trainLen + 1])
    # print(f'nrmse : {nrmse}')
    difference = prediction[0:errorLen] - data_tensor[trainLen + 1: trainLen + errorLen + 1, :]
    # print(f'difference : {difference}')

    # Step 2: Square the differences
    squared_difference = difference ** 2
    # print(f'sequed_differnse : {squared_difference}')

    # Step 3: Calculate the mean of the squared differences
    mse = torch.mean(squared_difference)
    print(f'mse : {mse}')

    # Step 2: Calculate RMSE
    rmse = torch.sqrt(mse)

    range_ = (torch.max(data_tensor[trainLen + 1: trainLen + errorLen + 1, :]) - torch.min(data_tensor[trainLen + 1: trainLen + errorLen + 1, :]))

    # Step 3: Normalize by range
    nrmse_range = rmse / range_
    #------------------------------------------------------
    weights = torch.linspace(1.0, 0.1, steps=errorLen)
    weighted_error = weights * torch.abs(prediction[0:errorLen] - data_tensor[trainLen + 1: trainLen + errorLen + 1, :])
    weighted_mae = torch.mean(weighted_error)
    # print(".................> ",nrmse_range)

    return nrmse_range









res_size = 1000
np.random.seed(42)
numpy_array1 = np.random.rand(res_size, 1 + 1)
numpy_array2 = (np.random.rand(res_size, res_size) -0.5)
# print(f'W : {numpy_array2}')
x = torch.tensor(numpy_array2 , requires_grad=True, dtype=torch.float64)
x_main = x
x_new = None
learning_rate = 1e-4
optimizer = optim.Adam([x], lr=learning_rate)  # Lowering the learning rate
list_of_loss = []
last_value = 0
number_epoch = 100

history_x = []
for epoch in range(number_epoch):  # Adjust number of epochs as needed
    print(f' number of epoch : {epoch}')
    optimizer.zero_grad()  # Zero the gradients

    # torch.nn.utils.clip_grad_norm_([x], max_norm=0.5)
    nrmse = loss_func(x)  # Compute the loss
    # make_dot(nrmse, params={"x": x}).render("loss_graph", format="png")
    history_x.append(x.clone().detach().tolist())
    nrmse.backward()  # Backpropagate
    # torch.nn.utils.clip_grad_norm_([x], max_norm=0.5)
    optimizer.step()  # Update the variable
    print(f'NRMSE = {nrmse}')
    # print(f'x = {x}')
    print("x grad : ", x.grad)
    # print(f'After step: x = {x.item()}')
    # x_new = x.item()
    list_of_loss.append(nrmse.item())
    if last_value < nrmse:
        print("  +   +   +   +")
    if last_value > nrmse:
        print(" -  -  -  -")
    if last_value == nrmse:
        print(" =  =  =  =")
    last_value = nrmse
    # print(f'x_new = {x_new} ')
    # print(f'Epoch {epoch+1}: res size = {x.item()}, Loss = {loss.item()}')
file_name = f'loss_backpropagation_on_reservoir_tue={tau}_NE={number_epoch}_NR={res_size}_lr={learning_rate}_sigmoid'
# Plot the loss curve
plt.plot(range(len(list_of_loss)), list_of_loss)
plt.xlabel('Epoch')
plt.ylabel('NRMSE')
plt.title('rmse vs Epoch')
plt.savefig(f'{file_name}.png')
plt.show()
print(f' if {history_x[-1]==x_main} ????')
print(f'range of loss : {list_of_loss[0]} - {list_of_loss[-1]}')

# sequenceLength = 10000
# tau = 29
# dimensions = 1
# inputSequence, outputSequence = generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions)
# print("Input Sequence Shape:", inputSequence.shape)
# print("Output Sequence Shape:", outputSequence.shape)

data = outputSequence[14000:]
input_size = output_size = 1
# res_size =
initLen = 100
errorLen = 200
num_reservoirs = 1
a = 0.3
reg = 4.840e-07
pre_train = 0.2
pre_test = 0.8
len_of_data = len(data)
trainLen = int(pre_train * len_of_data)
testLen = len_of_data - trainLen

torch.set_default_dtype(torch.double)

np.random.seed(42)
numpy_array1 = np.random.rand(res_size, 1 + input_size)

Win = nn.Parameter((torch.tensor(numpy_array1) - 0.5) * 1)
W = torch.tensor(history_x[-1])
W = W.reshape((res_size, res_size))

data_test = data[trainLen:]
data_train = data[:trainLen]

torch.set_default_dtype(torch.double)
data = torch.tensor(data)

esn = ESN(input_size, res_size, output_size, a, reg, W)
prediction = esn(data, trainLen, testLen, initLen, errorLen)
# print(f'  prediction ===================== {prediction}')
data_tensor = torch.tensor(data)
print(f'prediction[:, 0:errorLen] : {prediction[0:errorLen].shape}')
print((f'data_tensor[trainLen + 1: trainLen + errorLen + 1, :] : {data_tensor[trainLen + 1: trainLen + errorLen + 1, :].shape}'))
difference = prediction[0:errorLen] - data_tensor[trainLen + 1: trainLen + errorLen + 1, :]
squared_difference = difference ** 2
mse = torch.mean(squared_difference)
print(f'mse : {mse}')
rmse = torch.sqrt(mse)
range_ = (torch.max(data_tensor[trainLen + 1: trainLen + errorLen + 1, :]) - torch.min(data_tensor[trainLen + 1: trainLen + errorLen + 1, :]))
nrmse_range = rmse / range_
print(f'nrmse test : {nrmse_range}')
pred  =  prediction[0:errorLen]
actual =  data_tensor[trainLen + 1: trainLen + errorLen + 1, :]
plt.figure(figsize=(12, 6))

plt.plot(pred.detach().numpy(), label='Predicted test',color='red')
plt.plot(actual.detach().numpy(), label='Actual test',color = 'blue')
plt.title('test')
plt.savefig(f'{file_name}_test.png')
plt.show()

end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
time_string = f'Elapsed time: {elapsed_time} seconds'
print(time_string)
# Open a file in write mode
with open(f"{file_name}.txt", "w") as file:
    # Write the desired text to the file
    file.write(f"loss nrmse of test data : {nrmse_range} \n loss mse of test data : {mse} \nrange of optimize nrmse in train [ {list_of_loss[0]} - {list_of_loss[-1]} ] \ntime of running code : {time_string}")



