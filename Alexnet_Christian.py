from mmap import mmap
import numpy as np
import torch as torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import warnings
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Define layers of AlexNet

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = torch.nn.Linear(in_features = 9216, out_features=4096)
        self.linear2 = torch.nn.Linear(in_features = 4096, out_features=4096)
        self.linear3 = torch.nn.Linear(in_features = 4096, out_features=4)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU(inplace=True)

        # Added regularization layers

        self.normal1 = nn.BatchNorm2d(96)
        self.normal2 = nn.BatchNorm2d(256)
        self.normal3 = nn.BatchNorm2d(384)
        self.normal4 = nn.BatchNorm2d(384)
        self.normal5 = nn.BatchNorm2d(256)
        self.normal6 = nn.BatchNorm1d(4096)
        self.normal7 = nn.BatchNorm1d(4096)

    def forward(self, input):

        # Alexnet Convolutional Module

        input = self.conv1(input)
        input = self.normal1(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv2(input)
        input = self.normal2(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv3(input)
        input = self.normal3(input)
        input = self.relu(input)
        input = self.conv4(input)
        input = self.normal4(input)
        input = self.relu(input)
        input = self.conv5(input)
        input = self.normal5(input)
        input = self.relu(input)
        input = self.maxpool(input)

        # Flatten data before fc layer

        input = torch.flatten(input, 1)

        # Alexnet FC Module

        input = self.linear1(input)
        input = self.normal6(input)
        input = self.relu(input)
        input = self.dropout(input)
        input = self.linear2(input)
        input = self.normal7(input)
        input = self.relu(input)
        input = self.dropout(input)
        input = self.linear3(input)
    
        return input

net = Network()
net = net.float()

# Weights for convolutional and fc layers are initialized using the xavier normal method

net.conv1.weight = torch.nn.init.xavier_normal_(net.conv1.weight)
net.conv2.weight = torch.nn.init.xavier_normal_(net.conv2.weight)
net.conv3.weight = torch.nn.init.xavier_normal_(net.conv3.weight)
net.conv4.weight = torch.nn.init.xavier_normal_(net.conv4.weight)
net.conv5.weight = torch.nn.init.xavier_normal_(net.conv5.weight)
net.linear1.weight = torch.nn.init.xavier_normal_(net.linear1.weight)
net.linear2.weight = torch.nn.init.xavier_normal_(net.linear2.weight)
net.linear3.weight = torch.nn.init.xavier_normal_(net.linear3.weight)

# Device is defined

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Optimizer, scheduler, loss function, and variables such as lr, weight decay, training value size, batch size, and number of epochs are defined

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.0001)    # This lr and weight decay were determined to optimize test accuracy
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)                      # Constant LR scheduler was seen as good enough
loss_func = torch.nn.CrossEntropyLoss()                                         # Cross Entropy Loss was chosen for this model
train_val_split = 0.8                                                           # 0.8 Ensures that there is plenty of training data
batch_size = 20                                                                 # This batch size optimized accuracy
epochs = 100                                                                    # 100 Epochs was all that was needed to reach test accuracy peak

# Here the data is loaded, converted to a pytorch tensor, organized, and reduced to a 227 image size

data_X = np.load('X_train.npy')
data_y = np.load('y_train.npy')
data_X = torch.from_numpy(data_X)
data_y = torch.from_numpy(data_y)
data_X = torch.permute(data_X, (0, 3, 1, 2))
data_X = torch.nn.functional.interpolate(data_X, size=227)

# Training set provided is divided into both training and test sets

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=1-train_val_split)

# The test set provided is loaded, converted to a pytorch tensor, organized, and reduced to a 227 image size

x_test_final = np.load('X_test.npy')
x_test_final = torch.from_numpy(x_test_final)
x_test_final = torch.permute(x_test_final, (0, 3, 1, 2))
x_test_final = torch.nn.functional.interpolate(x_test_final, size=227)

# If possible, all data and the model itself are transferred to the GPU for faster processing

net = net.to(device=device)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
x_test_final = x_test_final.to(device)

# The training set is augmented and the augmented data is added to the unaugmented data
# Only two transforms were used as computer memory was becoming an issue

transform_1 = transforms.RandomHorizontalFlip(p=1)
transform_2 = transforms.RandomVerticalFlip(p=1)

X_train_transform_1 = transform_1(X_train)
X_train_transform_2 = transform_2(X_train)

X_train = torch.cat((X_train, X_train_transform_1, X_train_transform_2), 0)
y_train = torch.cat((y_train, y_train, y_train), 0)

# Class weights are tallied and used to define a sampler

class_sample_count = np.array([len(np.where(y_train.cpu() == t)[0]) for t in np.unique(y_train.cpu())])
weight = 1./class_sample_count
samples_weight = torch.from_numpy(np.array([weight[int(t)] for t in y_train]))
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight))

# Data loader with batches and sampling is defined

trainDataset = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=False, sampler=sampler)

net.train()

max_model_accuracy = 0

# Initialize arrays for graphing

epoch_array = []
training_array = []
test_array = []
f1_array = []

for i in range(epochs):
    total_loss = 0
    total_images = 0
    total_correct = 0
    current_max = False

    for image, label in trainloader:

        # Run train data through model and analyze loss and accuracy

        prediction = net(image.float())
        loss = loss_func(prediction, torch.tensor(label, dtype=torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = prediction.argmax(dim=1)
        total_loss += loss.item()
        total_images += label.size(0)
        total_correct += output.eq(label).sum().item()
        _, predicted = torch.max(prediction.data, 1)

    # Present accuracy to console

    model_accuracy = total_correct / total_images * 100
    print('Epoch: {0}, Loss: {1:.2f}, Train Accuracy {2:.2f}%'.format(i+1, total_loss, model_accuracy), end=', ')

    training_array.append(model_accuracy/100)

    # Perform testing here

    net.eval()

    with torch.no_grad():

        # Run test data through model and analyze loss and accuracy

        total_images = 0
        total_correct = 0
        prediction = net(X_test.float())
        _, predicted = torch.max(prediction.data, 1)
        total_images += y_test.size(0)
        total_correct += (predicted == y_test).sum().item()
        model_accuracy = total_correct / total_images * 100
        f1_score_instance = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')

        # For providing final predictions for submission
        # Check if highest test accuracy was achieved
        # If so, create a new prediction
        # Ensured that the prediction submitted was during test accuracy peak and not just end of epochs

        if model_accuracy > max_model_accuracy:
            max_model_accuracy = model_accuracy
            current_max = True
            np.savetxt('Prediction_Christian.npy', np.array(net(x_test_final.float()).cpu().argmax(dim=1)))

        # Present accuracy to console

        print('Test Accuracy: {0:.2f}%, F1 Score: {1:.2f}, Current Maximum: {2}'.format(model_accuracy, f1_score_instance, current_max))

    # Update next element of array graphs

    epoch_array.append(i)
    test_array.append(model_accuracy/100)
    f1_array.append(f1_score_instance)
    net.train()

# Plot final arrays

plt.plot(epoch_array, training_array, color='r')
plt.plot(epoch_array, test_array, color='g')
plt.plot(epoch_array, f1_array, color='b')
plt.legend(['Training', 'Test', 'F1 Score'])
plt.title('AlexNet Analysis')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()