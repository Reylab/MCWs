# Author: Sunil Mathew
# Date: 1 April 2024
# RasterNet: Deep Neural Network based classfication of rasters into good, bad, and meh classes

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchviz
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from collections import Counter

class RasterDataset(Dataset):
    """
    A custom dataset class for handling raster data.

    Args:
        root_dir (str): The root directory containing the raster data.

    Attributes:
        root_dir (str): The root directory containing the raster data.
        classes (dict): A dictionary mapping class names to class labels.
        data (list): A list to store the raster data.
        labels (list): A list to store the corresponding labels.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the raster image and its label at the given index.

    """

    def __init__(self, root_dir, raster_size=150):
        self.root_dir = root_dir
        self.current_n_trials = 6
        self.raster_time_win = 3000 # ms
        self.time_pre = 1000 # ms
        self.raster_size = raster_size
        self.dl_resol = self.raster_time_win / self.raster_size
        self.classes = {'bad' : 1, 'good' : 0, 'meh' : 2}
        self.data = {3: [], 6: [], 9: []}
        self.labels = {3: [], 6: [], 9: []}
        self.valid_data = {3: False, 6: False, 9: False}
        for path, dirs, files in os.walk(root_dir):
            for dir in dirs:
                for file in os.listdir(os.path.join(path, dir)):
                    class_lbl = os.path.basename(path)
                    if file.endswith('.npy'):
                        raster = np.load(os.path.join(path, dir, file), allow_pickle=True)
                        if raster.size < 150:
                            raster_img = self.convert_to_raster_img(raster)
                        else:
                            raster_img = raster

                        if type(raster_img) == np.ndarray and raster_img.shape[1] != raster_size:
                            continue
                        if raster_img.shape[0] not in self.data.keys():
                            self.data[raster_img.shape[0]] = []
                            self.labels[raster_img.shape[0]] = []
                        for key in self.data.keys():
                            if raster_img.shape[0] > key:
                                subraster = raster_img[:key, :]
                                self.data[key].append(subraster)
                                self.labels[key].append(self.classes[class_lbl])
                        self.data[raster_img.shape[0]].append(raster_img)
                        self.labels[raster_img.shape[0]].append(self.classes[class_lbl])

        # Class distribution before SMOTE
        self.class_distribution(plot=False)

        # Perform SMOTE oversampling to balance the classes
        self.smote_oversampling()
        
        # Class distribution after SMOTE
        self.class_distribution(plot=False)

    def convert_to_raster_img(self, raster):
        # Locals
        resp_dl = []

        for resp in raster:
            response = np.zeros(self.raster_size)
            if type(resp) == np.ndarray:
                for spike in resp:
                    spike_idx = int((spike+self.time_pre)/self.dl_resol) - 1
                    response[spike_idx] = 1
            elif resp == None:
                pass
            else:
                spike_idx = int((resp+self.time_pre)/self.dl_resol) - 1
                if spike_idx < self.raster_size and spike_idx >= 0:
                    response[spike_idx] = 1
            resp_dl.append(response)

        return np.array(resp_dl)

    def class_distribution(self, plot=True):
        """
        Prints the distribution of classes in the dataset.
        """
        for key in self.labels.keys():
            print(f'Class distribution for {key} rasters: {Counter(self.labels[key])}')    

        if plot:
            # create bar plot
            fig, ax = plt.subplots(1, len(self.labels.keys()), figsize=(15, 5))
            for i, key in enumerate(self.labels.keys()):
                ax[i].bar(self.classes.keys(), Counter(self.labels[key]).values())
                ax[i].set_title(f'{key} trials')
                ax[i].set_xlabel('Classes')
                ax[i].set_ylabel('Frequency')
            plt.show()        
        

    def smote_oversampling(self):
        """
        Performs SMOTE oversampling to balance the classes.
        """
        for key in self.data.keys():
            data = np.array(self.data[key])
            labels = np.array(self.labels[key])
            smote = SMOTE()
            try:
                data_sm, labels_sm = smote.fit_resample(data.reshape(len(data), -1), labels)
                self.data[key] = data_sm.reshape(-1, data.shape[1], data.shape[2])
                self.labels[key] = labels_sm
                self.valid_data[key] = True
            except:
                print(f'Could not perform SMOTE oversampling for {key} rasters')
                self.valid_data[key] = False

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.data[self.current_n_trials])

    def __getitem__(self, idx):
        """
        Returns the raster image and its label at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the raster image and its label.

        """
        image, label = self.data[self.current_n_trials][idx], self.labels[self.current_n_trials][idx]
        return torch.tensor(image).float(), torch.tensor(label).long()

# Define the CNN model to classify rasters (6x150) into three classes good, bad, and meh
# The model ouputs a 3x1 tensor with the probabilities of each class
class RasterFCN(nn.Module):
    """
    RasterFCN is a PyTorch module for a fully connected neural network with softmax activation.
    It takes a 2D input tensor and performs forward propagation to produce a 3-class output.
    """

    def __init__(self, in_size):
        super(RasterFCN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(in_size, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class RasterNet():
    """
    RasterNet class represents a neural network model for raster image classification.

    Attributes:
        model (RasterFCN): The underlying neural network model.
    """

    def __init__(self, c, n_trials=6, raster_size=150):
        """
        Initializes a new instance of the RasterNet class.
        """
        self.c = c # Communication class for things like progress updates
        self.model = RasterFCN(in_size=n_trials * raster_size)
        self.raster_classes = {0: 'good', 1: 'bad', 2: 'meh'}
        self.n_trials = n_trials

    def train_model(self, train_data, batch_size=32, num_epochs=10, verbose=False):
        """
        Trains the RasterNet model using the provided training data.

        Args:
            train_data (list): The training data, a list of tuples containing input images and labels.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            num_epochs (int, optional): The number of epochs to train for. Defaults to 10.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(0, len(train_data), batch_size):
                inputs_batch = []
                labels_batch = []
                for j in range(batch_size):
                    if i + j >= len(train_data):
                        break
                    inputs, labels = train_data[i + j]
                    inputs_batch.append(inputs.unsqueeze(0))
                    labels_batch.append(labels)
                inputs_batch = torch.cat(inputs_batch, dim=0)
                labels_batch = torch.tensor(labels_batch)

                optimizer.zero_grad()
                outputs = self.model(inputs_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0 and verbose:
                    msg = f'[Epoch {epoch+1}/{num_epochs}, {i+1:5d}/{len(train_data)}] loss: {running_loss/100:.3f}'
                    print(msg)
                    # self.c.progress.emit(int((epoch + 1) / num_epochs * 100), msg)
                    running_loss = 0.0

        print('Finished Training')

    def save_model(self, path):
        """
        Saves the trained model to the specified path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def test_model(self, test_data):
        """
        Tests the trained model using the provided test data and prints the accuracy.

        Args:
            test_data (list): The test data, a list of tuples containing input images and labels.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                outputs = self.model(images.unsqueeze(0))
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += (predicted == labels).sum().item()

        # print(f'Accuracy of the network on the test images({correct}/{total}): {(100 * correct / total):.2f}%')
        return correct, total

    def plot_predictions(self, test_data):
        """
        Plots a grid of rasters and their predictions with probabilities.

        Args:
            test_data (list): The test data, a list of tuples containing input images and labels.
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 3)
        fig.suptitle('Raster predictions')
        
        for i in range(3):
            for j in range(3):
                image, label = test_data[i * 3 + j]
                output = self.model(image.unsqueeze(0))
                prob, predicted = torch.max(output.data, 1)
                axs[i, j].imshow(image, cmap='gray')
                axs[i, j].set_title(f'True: {self.raster_classes[label.numpy().item()]} ' + 
                                    f'Predicted: {self.raster_classes[predicted.numpy()[0]]} ' +
                                    f'(Prob: {prob.numpy()[0] * 100:.2f}%)')
                axs[i, j].axis('off')

        plt.show()

    def predict_raster_class(self, image):
        """
        Predicts the class of a single raster image.
        """
        output = self.model(image.unsqueeze(0))
        prob, predicted = torch.max(output.data, 1)
        return prob.numpy()[0], self.raster_classes[predicted.numpy()[0]]
    
    def plot_model(self, n_trials):
        """
        Plots the model architecture.
        """
        x = torch.randn(1, n_trials, 150)

        # Save the plot to a file
        torchviz.make_dot(self.model(x), 
                          params=dict(self.model.named_parameters())).render("model", format="png")

if __name__ == '__main__':
    raster_size_px = 150
    b_load_model = False
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rasternet')
    data_folder = os.path.join(root_dir, 'data')
    models_folder = os.path.join(root_dir, 'models')
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    # Set up data loader
    raster_dataset = RasterDataset(root_dir=data_folder, raster_size=raster_size_px)
    for key in raster_dataset.data.keys():
        raster_net = RasterNet(c=None, n_trials=key, raster_size=raster_size_px)
        model_path = os.path.join(models_folder, f'rasternet_{key}_{raster_size_px}.pt')
        if b_load_model and os.path.exists(model_path):
            raster_net.model.load_state_dict(torch.load(model_path))
            # raster_net.plot_model(n_trials=key)
        else:
            if os.path.exists(model_path):
                # Show warning that model already exists and ask if user wants to overwrite in command line
                inp = input(f'Model for {key} rasters already exists. Overwrite? (y/n): ')
                if inp.lower() == 'n':
                    continue

            if raster_dataset.valid_data[key]:
                raster_dataset.current_n_trials = key
                train_data, test_data = torch.utils.data.random_split(raster_dataset, 
                                                                    [int(0.8 * len(raster_dataset)), 
                                                                    len(raster_dataset) - int(0.8 * len(raster_dataset))])
                raster_net.train_model(train_data, batch_size=32, num_epochs=10)
                raster_net.save_model(model_path)
                corr, tot = raster_net.test_model(test_data)
                print(f'Accuracy of {key} trial model (sz:{raster_size_px}) ({corr}/{tot}): {(100 * corr / tot):.2f}%')
                # raster_net.plot_predictions(test_data)

    

