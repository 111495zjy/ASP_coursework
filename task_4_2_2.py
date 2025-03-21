import random
random.seed(0)
import torch
import torch.nn as nn
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
import numpy as np
np.random.seed(0)
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dct = scipy.io.loadmat('C:\\Users\\zjy16\\Desktop\\adaptive\\4\\ASPMI_CW42_Data.mat')
# Extract training data and labels
trainX = data_dct['trainX']
trainy = data_dct['trainy']
trainX, trainy = shuffle(trainX, trainy)

# Define training and model parameters
num_epochs = 200
batch_size = 30
n_in, n_out = 1, 8
kernel_size = 500
padding = 0
dilation = 1
stride = 1
learning_rate = 0.000001
L_in = trainX.shape[-1]
trainX = trainX.reshape((trainX.shape[0], 1, L_in))

# Calculate the output length after convolution
L_out = int((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

# Define the model using pytorch
class ConvNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=padding)
        self.layer2 = nn.MaxPool1d(L_out)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.abs(out)
        out = self.layer2(out)
        out = self.act(out)
        return out

model = ConvNet1D().to(device)

# Loss and optimizer
criterion = nn.NLLLoss()  # nn.BCEWithLogitsLoss() # nn.L1Loss() # nn.MSELoss() # BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = trainX.shape[0]

# Transformation of data into torch tensors
trainXT = torch.from_numpy(trainX.astype('float32')).to(device)
trainyT = torch.from_numpy(trainy.astype('float32')).to(device)

##### Create initialized weights for each of the 8 kernels
# Initialize layer weights as zeros for each of the 8 layers
# Nt = kernel_size
layer1_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer2_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer3_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer4_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer5_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer6_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer7_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))
layer8_init = np.reshape(np.zeros((1, kernel_size)), (1, 1, kernel_size))

# Concatenate all layers into one array
weights_init = np.concatenate((layer1_init, layer2_init, layer3_init, layer4_init,
                               layer5_init, layer6_init, layer7_init, layer8_init), axis=0)

# Convert the numpy weights array to PyTorch tensor and set to model's layer1 weights
weights_initT = torch.from_numpy(weights_init.astype('float32')).to(device)
model.layer1.weight.data = weights_initT

# Loss list to store the training loss per epoch
loss_list = []

# Training loop
for epoch in range(num_epochs):
    correct_sum = 0
    correct_sum_test = 0
    for i in range(total_step // batch_size):  # split data into batches
        trainXT_seg = trainXT[i * batch_size:(i + 1) * batch_size, :, :]
        trainyT_seg = trainyT[i * batch_size:(i + 1) * batch_size, :]
        
        # Get the target indices (the label classes)
        target_indices = torch.argmax(trainyT_seg, dim=1)
        
        # Run the forward pass
        outputs = model(trainXT_seg)
        
        # Ensure outputs are the correct shape
        outputs = outputs.reshape((outputs.shape[0], 8))
        
        # Calculate loss for the batch
        loss = criterion(outputs, target_indices)
        
        # Backpropagation and Adam optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        output_max = torch.argmax(outputs, 1, keepdim=False)
        reference_max = torch.argmax(trainyT_seg, 1, keepdim=False)
        count = torch.count_nonzero((output_max - reference_max)).cpu().detach()
        accuracy = 100 * ((batch_size - count) / batch_size)
        
        # Print out the loss and accuracy for each epoch
        print("Epoch:", epoch)
        print("Train loss:", loss.item())
        print("Accuracy (%):", accuracy.item())
        
        # Append the loss to the loss list
        loss_list.append(loss.item())

# Plot the training loss per epoch
plt.title('Training loss')
plt.ylabel('Negative Log-likelihood Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(loss_list, label="Training loss")
plt.legend()
plt.show()
#### plot the trained kernel weights
kernels = model.layer1.weight.T.cpu().detach().numpy()[:,0,:]
fig, axs = plt.subplots(kernels.shape[1],1, figsize=(5,8), squeeze=True)
for i, ax in enumerate(axs):
    ax.plot(kernels[:,i])
plt.show()