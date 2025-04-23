#!/usr/bin/env python
# coding: utf-8

# In[52]:

print("Please Work")
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from PIL import Image
import random
import torch
from torch import autograd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
output_dir = '/home-mscluster/glaxton/TIL'
print("Imports loaded")

# Define the transform for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load and transform the datasets
transform = transforms.Compose([transforms.ToTensor()])

# Standard MNIST for Task A (Digit Classification)
train_A_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_A_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Function to create a balanced subset
def create_balanced_subset(dataset, num_samples_per_class):
    class_indices = {i: [] for i in range(10)}  # Store indices for each class (0-9)

    # Sort dataset into classes
    for idx, (_, target) in enumerate(dataset):
        class_indices[target].append(idx)

    # Randomly select num_samples_per_class indices per class
    subset_indices = []
    for class_label, indices in class_indices.items():
        subset_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

    # Create the subset
    subset = Subset(dataset, subset_indices)
    return subset

# Create a subset with 100 samples per digit
train_A_dataset = create_balanced_subset(train_A_dataset, num_samples_per_class=1500)
test_A_dataset = create_balanced_subset(test_A_dataset, num_samples_per_class=150)



# Custom dataset for Task B (Even-Odd Classification)
class EvenOddMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        img, target = self.mnist_dataset[index]
        even_odd_target = target % 2  # 0 for even, 1 for odd
        return img, even_odd_target

    def __len__(self):
        return len(self.mnist_dataset)

train_B_dataset = EvenOddMNIST(train_A_dataset)
test_B_dataset = EvenOddMNIST(test_A_dataset)

class TaskIdentifierDataset(Dataset):
    def __init__(self, dataset, task_id):
        self.dataset = dataset
        self.task_id = task_id

    def __getitem__(self, index):
        img, target = self.dataset[index]
        task_identifier = torch.tensor([self.task_id], dtype=torch.float32)
        return img, target, task_identifier

    def __len__(self):
        return len(self.dataset)

# Wrapping the original datasets with task identifiers
train_A_dataset_with_id = TaskIdentifierDataset(train_A_dataset, 1.0)
test_A_dataset_with_id = TaskIdentifierDataset(test_A_dataset, 1.0)
train_B_dataset_with_id = TaskIdentifierDataset(train_B_dataset, 0.0)
test_B_dataset_with_id = TaskIdentifierDataset(test_B_dataset, 0.0)

# Split into training and validation sets for Task A and Task B
train_size_A = int(0.8 * len(train_A_dataset_with_id))  # 80% for training
val_size_A = len(train_A_dataset_with_id) - train_size_A  # 20% for validation

train_size_B = int(0.8 * len(train_B_dataset_with_id))  # 80% for training
val_size_B = len(train_B_dataset_with_id) - train_size_B  # 20% for validation

train_A_dataset_with_id, val_A_dataset_with_id = random_split(train_A_dataset_with_id, [train_size_A, val_size_A])
train_B_dataset_with_id, val_B_dataset_with_id = random_split(train_B_dataset_with_id, [train_size_B, val_size_B])

# Create DataLoaders
train_A_loader = DataLoader(train_A_dataset_with_id, batch_size=64, shuffle=True)
val_A_loader = DataLoader(val_A_dataset_with_id, batch_size=64, shuffle=False)
test_A_loader = DataLoader(test_A_dataset_with_id, batch_size=64, shuffle=False)

train_B_loader = DataLoader(train_B_dataset_with_id, batch_size=64, shuffle=True)
val_B_loader = DataLoader(val_B_dataset_with_id, batch_size=64, shuffle=False)
test_B_loader = DataLoader(test_B_dataset_with_id, batch_size=64, shuffle=False)


# # In[299]:


# Check the size of the subset
print(f"train_A_dataset size: {len(train_A_dataset)}")
print(f"train_B_dataset size: {len(train_B_dataset)}")
print(f"test_B_dataset size: {len(test_B_dataset)}")
print(f"test_A_dataset size: {len(test_A_dataset)}")
# In[54]:

class ConvNet(nn.Module):
    def __init__(self, shared_dim=128, output_dim_A=10, output_dim_B=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28 + 1, shared_dim)  # Adjusted to match the output size after conv layers + task identifier
        self.fc2_A = nn.Linear(shared_dim, output_dim_A)
        self.fc2_B = nn.Linear(shared_dim, output_dim_B)

    def forward(self, x, task_id):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        task_id = task_id.view(-1, 1)  # Ensure task_id has shape (batch_size, 1)
        x = torch.cat((x, task_id), dim=1)  # Concatenate task identifier
        x = F.relu(self.fc1(x))

        # Generate outputs for both tasks
        output_A = self.fc2_A(x)
        output_B = self.fc2_B(x)

        # Combine the outputs into a single tensor with proper shape
        output = torch.zeros(x.size(0), max(output_A.size(1), output_B.size(1)), device=x.device)
        output[:, :output_A.size(1)] = output_A * (task_id == 1).float()
        output[:, :output_B.size(1)] += output_B * (task_id == 0).float()

        return output


# Training function
def train(model, loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target, task_id in loader:  # Ensure the DataLoader returns three items
            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader)}')

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target, task_id in loader:  # Ensure the DataLoader returns three items
            output = model(data, task_id)
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            all_targets.extend(target.numpy())
            all_predictions.extend(predictions.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_predictions)
    
    return accuracy, precision, recall, kappa


# In[55]:


def train_with_patience(model, loader, optimizer, criterion, epochs=20, patience=5):
    model.train()
    best_loss = float('inf')
    no_improvement = 0
    epoch_accuracy = []
    epochs_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for data, target, task_id in loader:  # Ensure the DataLoader returns three items
            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total
        epoch_accuracy.append(accuracy)
        epochs_loss.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
        else:
            no_improvement += 1

        # Early stopping criteria
        if no_improvement >= patience:
            print(f"Stopping early due to no improvement in loss for {patience} consecutive epochs.")
            break

    return epoch_accuracy, epochs_loss


# In[56]:


import torch
from torch import autograd

import torch
import torch.nn.functional as F

class OnlineEWC:
    def __init__(self, model, dataloader):
        """
        Online EWC implementation with task-aware Fisher matrices.

        Args:
            model: Neural network model.
            dataloader: DataLoader for the current task.
        """
        self.model = model
        self.dataloader = dataloader
        self.fisher_matrix = {}  # Dictionary storing Fisher matrices per task
        self.optimal_params = {}

        # Initialize Fisher matrix and optimal parameters
        self._initialize_params()

    def _initialize_params(self):
        """
        Initialize or reset Fisher matrix and optimal_params to match the current model's parameters.
        """
        self.fisher_matrix = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def update_fisher_matrix(self):
        """
        Update the Fisher matrix using the online method.
        """
        self.model.eval()
        fisher_diagonal = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        dataset_size = len(self.dataloader.dataset)  # Dataset size for normalization
        
        for i, (data, target, task_id) in enumerate(self.dataloader, 1):
            self.model.zero_grad()
            output = self.model(data, task_id)

            # Calculate loss using CrossEntropyLoss
            loss = F.cross_entropy(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_diagonal[n] += (p.grad ** 2 - fisher_diagonal[n]) / i

        # Normalize and store Fisher information
        for n in fisher_diagonal:
            if n in self.fisher_matrix:
                self.fisher_matrix[n] += fisher_diagonal[n] / max(1e-10, fisher_diagonal[n].norm().item())
            else:
                self.fisher_matrix[n] = fisher_diagonal[n] / max(1e-10, fisher_diagonal[n].norm().item())

    def store_optimal_params(self):
        """
        Store the current model parameters as the optimal parameters after training a task.
        """
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        """
        Calculate the EWC penalty based on the Fisher matrix and optimal parameters.

        Args:
            model: The current model.

        Returns:
            penalty: EWC regularization term.
        """
        penalty = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher_matrix and n in self.optimal_params:
                diff = p - self.optimal_params[n]
                term = (self.fisher_matrix[n] * diff ** 2).sum()
                penalty += term
        return penalty






# In[57]:


import torch
import torch.nn as nn
import torch.nn.functional as F

def train_with_ewc(model, optimizer, dataloaders, ewc=None, lambda_ewc=0.1, epochs=5):
    """
    Train the model with or without EWC penalty.

    Args:
        model: Neural network model to train.
        optimizer: Optimizer for training.
        dataloaders: Dictionary containing 'train' and optionally 'val' dataloaders.
        ewc: Instance of the EWC class, or None if not using EWC.
        lambda_ewc: Importance of the EWC penalty (default is 0.1).
        epochs: Number of epochs to train.
    """
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_penalty = 0.0

        for inputs, targets, task_id in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(inputs, task_id)
            loss = criterion(outputs, targets)

            # Add EWC penalty if applicable
            if ewc is not None:
                ewc_penalty = ewc.penalty(model)
                loss += lambda_ewc * ewc_penalty
                total_penalty += ewc_penalty.item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log losses
        avg_loss = running_loss / len(dataloaders['train'])
        avg_penalty = total_penalty / len(dataloaders['train']) if ewc is not None else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Penalty: {avg_penalty:.4f}")






# In[59]:


class SynapticIntelligence:
    def __init__(self, model, dataloader, importance=1000, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.importance = importance
        self.device = device
        self.saved_params = {}
        self.omega = {}

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.saved_params[n] = p.clone().detach().to(self.device)
                self.omega[n] = torch.zeros_like(p).to(self.device)

    def update_omega(self, batch_loss, lr):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.omega[n] += p.grad * (p.detach() - self.saved_params[n])
                    self.saved_params[n] = p.clone().detach()

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                _loss = self.omega[n] * (p - self.saved_params[n]) ** 2
                loss += _loss.sum()
        return self.importance * loss

    def end_task(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] /= len(self.dataloader)


# In[60]:


def train_with_si(model, loader, si, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target, task_id in loader:
            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            loss += si.penalty()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            si.update_omega(loss, optimizer.param_groups[0]['lr'])
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}')
    si.end_task()


# In[61]:


class RehearsalBuffer:
    def __init__(self, buffer_size=200):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_to_buffer(self, data, target, task_id):
        for i in range(len(data)):
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            self.buffer.append((data[i], target[i], task_id[i]))

    def get_buffer(self):
        data, target, task_id = zip(*self.buffer)
        return torch.stack(data), torch.tensor(target), torch.tensor(task_id)


# In[62]:


def train_with_rehearsal(model, loader, buffer, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target, task_id in loader:
            buffer.add_to_buffer(data, target, task_id)
            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            if len(buffer.buffer) > 0:
                buffer_data, buffer_target, buffer_task_id = buffer.get_buffer()
                buffer_output = model(buffer_data, buffer_task_id)
                loss += criterion(buffer_output, buffer_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}')


# In[63]:


class PNN(nn.Module):
    def __init__(self, shared_dim=128, output_dim_A=10, output_dim_B=2):
        super(PNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1_A = nn.Linear(64 * 28 * 28, shared_dim)
        self.fc2_A = nn.Linear(shared_dim, output_dim_A)
        self.fc1_B = nn.Linear(64 * 28 * 28, shared_dim)
        self.fc2_B = nn.Linear(shared_dim, output_dim_B)

    def forward(self, x, task_id):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if task_id[0] == 1:
            x = F.relu(self.fc1_A(x))
            output = self.fc2_A(x)
        else:
            x = F.relu(self.fc1_B(x))
            output = self.fc2_B(x)
        return output


# In[64]:


def train_pnn(model, loader, optimizer, criterion, task_num, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target, task_id in loader:
            optimizer.zero_grad()
            output = model(data, task_id)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}')

def evaluate_pnn(model, loader, criterion, task_num):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target, task_id in loader:
            output = model(data, task_id)
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            all_targets.extend(target.numpy())
            all_predictions.extend(predictions.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_predictions)
    
    return accuracy, precision, recall, kappa


# In[71]:


epoch_stats_other = {
    "si": {"acc_B": [], "loss_B": [], "task_A_during_B":[],"task_A_during_B_loss":[],"val_task_A_during_B":[],"val_acc_B":[]},
    "pnn": {"acc_B": [], "loss_B": [], "task_A_during_B":[],"task_A_during_B_loss":[],"val_task_A_during_B":[],"val_acc_B":[]},
    "reh": {"acc_B": [], "loss_B": [], "task_A_during_B":[],"task_A_during_B_loss":[],"val_task_A_during_B":[],"val_acc_B":[]}
}

num_runs = 10
epochs = 50


# In[72]:


# Initialize models, optimizers, and criteria
shared_dim = 128
output_dim_A = 10
output_dim_B = 2
lr = 0.00001

criterion = nn.CrossEntropyLoss()
import json

# Hyperparameters
learning_rate = 0.00001

# Initialize storage for results, including "Untrained_A"
results = {
    "untrained_A": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "initial_A": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "B_ewc": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "A_after_B_ewc": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "B_no_ewc": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "A_after_B_no_ewc": {"accuracy": [], "precision": [], "recall": [], "kappa": []}
}

results_val = {
    "initial_A_val": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "B_ewc_val": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "A_after_B_ewc_val": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "B_no_ewc_val": {"accuracy": [], "precision": [], "recall": [], "kappa": []},
    "A_after_B_no_ewc_val": {"accuracy": [], "precision": [], "recall": [], "kappa": []}
}

# Store per-epoch values for plotting
epoch_stats = {
    "Initial": {"accuracy": [], "loss": [],"task_B_during_A":[],"A_during_A":[],"A_during_A_val":[],"task_B_during_A_val":[]},
    "ewc": {"accuracy": [], "loss": [], "task_A_during_B":[], "val_task_A_during_B": [], "val_acc": [], "val_loss": [],"val_task_B":[],"task_B":[]},
    "no_ewc": {"accuracy": [], "loss": [], "task_A_during_B": [], "val_task_A_during_B": [], "val_acc": [], "val_loss": [],"task_B":[],"val_task_B":[]}
}

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")

    # Initialize the model for Task A
    modelA = ConvNet(shared_dim=128, output_dim_A=10, output_dim_B=2)
    optimizerA = optim.Adam(modelA.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Evaluate "Untrained_A" - performance on Task A before any training
    accuracy_untrained_A, precision_untrained_A, recall_untrained_A, kappa_untrained_A = evaluate(modelA, test_A_loader, criterion)
    print(f"Task A - Untrained: Accuracy: {accuracy_untrained_A}, Precision: {precision_untrained_A}, Recall: {recall_untrained_A}, Kappa: {kappa_untrained_A}")
    results["untrained_A"]["accuracy"].append(accuracy_untrained_A)
    results["untrained_A"]["precision"].append(precision_untrained_A)
    results["untrained_A"]["recall"].append(recall_untrained_A)
    results["untrained_A"]["kappa"].append(kappa_untrained_A)

    task_B_during_A = []
    task_A_during_A = []

    task_B_during_A_val = []
    task_A_during_A_val = []
    ewc = OnlineEWC(modelA, train_A_loader)

    # Train model on Task A (digit classification)
    for epoch in range(epochs):
        train_with_ewc(modelA, optimizerA, dataloaders = {'train': train_A_loader, 'val': val_A_loader} , ewc = None, lambda_ewc= 0, epochs = 1)

        # Track performance on Task B
        accuracy_taskA_during_A, _, _, _ = evaluate(modelA, test_A_loader, criterion)
        print(f"Epoch {epoch + 1} - Task A during A (with EWC): {accuracy_taskA_during_A:.4f}")
        task_A_during_A.append(accuracy_taskA_during_A)

        accuracy_taskA_during_A_val, _, _, _ = evaluate(modelA, val_A_loader, criterion)
        print(f"Epoch {epoch + 1} - Task A during A (with EWC): {accuracy_taskA_during_A_val:.4f}")
        task_A_during_A_val.append(accuracy_taskA_during_A_val)

        # Track performance on Task B
        accuracy_taskB_during_A, _, _, _ = evaluate(modelA, test_B_loader, criterion)
        print(f"Epoch {epoch + 1} - Task B during A (with EWC): {accuracy_taskB_during_A:.4f}")
        task_B_during_A.append(accuracy_taskB_during_A)

        # Track performance on Task B
        accuracy_taskB_during_A_val, _, _, _ = evaluate(modelA, val_B_loader, criterion)
        print(f"Epoch {epoch + 1} - Task B during A (with EWC): {accuracy_taskB_during_A_val:.4f}")
        task_B_during_A_val.append(accuracy_taskB_during_A_val)


    torch.save(modelA.state_dict(), 'taskA_model.pth')

    # Store epoch-wise results for plotting
    epoch_stats["Initial"]["A_during_A"].append(task_A_during_A)
    epoch_stats["Initial"]["task_B_during_A"].append(task_B_during_A)

    epoch_stats["Initial"]["A_during_A_val"].append(task_A_during_A_val)
    epoch_stats["Initial"]["task_B_during_A_val"].append(task_B_during_A_val)

    # Evaluate initial performance on Task A
    accuracy_taskA_initial, precision_taskA_initial, recall_taskA_initial, kappa_taskA_initial = evaluate(modelA, test_A_loader, criterion)
    print(f"Task A - Initial: Accuracy: {accuracy_taskA_initial}, Precision: {precision_taskA_initial}, Recall: {recall_taskA_initial}, Cohen's Kappa: {kappa_taskA_initial}")
    results["initial_A"]["accuracy"].append(accuracy_taskA_initial)
    results["initial_A"]["precision"].append(precision_taskA_initial)
    results["initial_A"]["recall"].append(recall_taskA_initial)
    results["initial_A"]["kappa"].append(kappa_taskA_initial)

    # Evaluate initial performance on Task A - validation set
    accuracy_taskA_initial_val, precision_taskA_initial_val, recall_taskA_initial_val, kappa_taskA_initial_val = evaluate(modelA, val_A_loader, criterion)
    print(f"Task A - Initial: Accuracy: {accuracy_taskA_initial_val:.4f}, Precision: {precision_taskA_initial_val:.4f}, Recall: {recall_taskA_initial_val:.4f}, Cohen's Kappa: {kappa_taskA_initial_val:.4f}")
    results_val["initial_A_val"]["accuracy"].append(accuracy_taskA_initial_val)
    results_val["initial_A_val"]["precision"].append(precision_taskA_initial_val)
    results_val["initial_A_val"]["recall"].append(recall_taskA_initial_val)
    results_val["initial_A_val"]["kappa"].append(kappa_taskA_initial_val)

    # Initialize EWC
    #9223372036854775807
    #99999999999999999
    # Store task_A_during_B as a list for this run
    task_A_during_B_ewc_run = []
    task_A_during_B_no_ewc_run = []

    task_B_ewc_run = []
    task_B_no_ewc_run = []

    val_task_B_ewc_run = []
    val_task_B_no_ewc_run = []

    val_task_A_during_B_ewc_run = []
    val_task_A_during_B_no_ewc_run = []

    ewc.update_fisher_matrix()
    ewc.store_optimal_params()

    
    for epoch in range(epochs):
        # Train model on Task B with EWC
        train_with_ewc(modelA, optimizerA, dataloaders = {'train': train_B_loader, 'val': val_B_loader}, ewc = None, lambda_ewc= 1000000, epochs = 1)
    
        # Store epoch-wise results for Task B
        #epoch_stats["ewc"]["accuracy"].append(acc_B_ewc)
        #epoch_stats["ewc"]["loss"].append(loss_B_ewc)

        # Track performance on Task B
        accuracy_taskB, _, _, _ = evaluate(modelA, test_B_loader, criterion)
        print(f"Epoch {epoch + 1} - Task B (with EWC): {accuracy_taskB:.4f}")
        task_B_ewc_run.append(accuracy_taskB)

        # Track performance on Task B
        val_accuracy_taskB, _, _, _ = evaluate(modelA, val_B_loader, criterion)
        print(f"Epoch {epoch + 1} - Task B - validation set (with EWC): {val_accuracy_taskB:.4f}")
        val_task_B_ewc_run.append(val_accuracy_taskB)
    
        # Track performance on Task A while training on Task B
        accuracy_taskA_during_B_ewc, _, _, _ = evaluate(modelA, test_A_loader, criterion)
        print(f"Epoch {epoch + 1} - Task A performance during Task B (with EWC): {accuracy_taskA_during_B_ewc:.4f}")
        task_A_during_B_ewc_run.append(accuracy_taskA_during_B_ewc)

        # Track performance on Task A while training on Task B
        val_accuracy_taskA_during_B_ewc, _, _, _ = evaluate(modelA, val_A_loader, criterion)
        print(f"Epoch {epoch + 1} - Task A performance during Task B - validation set (with EWC): {val_accuracy_taskA_during_B_ewc:.4f}")
        val_task_A_during_B_ewc_run.append(val_accuracy_taskA_during_B_ewc)
        
    epoch_stats["ewc"]["task_A_during_B"].append(task_A_during_B_ewc_run)
    epoch_stats["ewc"]["val_task_A_during_B"].append(val_task_A_during_B_ewc_run)
    epoch_stats["ewc"]["task_B"].append(task_B_ewc_run)
    epoch_stats["ewc"]["val_task_B"].append(val_task_B_ewc_run)

    # Evaluate performance on Task B with EWC
    accuracy_taskB_ewc, precision_taskB_ewc, recall_taskB_ewc, kappa_taskB_ewc = evaluate(modelA, test_B_loader, criterion)
    print(f"Task B with EWC: Accuracy: {accuracy_taskB_ewc}, Precision: {precision_taskB_ewc}, Recall: {recall_taskB_ewc}, Cohen's Kappa: {kappa_taskB_ewc}")
    results["B_ewc"]["accuracy"].append(accuracy_taskB_ewc)
    results["B_ewc"]["precision"].append(precision_taskB_ewc)
    results["B_ewc"]["recall"].append(recall_taskB_ewc)
    results["B_ewc"]["kappa"].append(kappa_taskB_ewc)

    # Evaluate on Task B - Validation
    accuracy_taskB_ewc_val, precision_taskB_ewc_val, recall_taskB_ewc_val, kappa_taskB_ewc_val = evaluate(modelA, val_B_loader, criterion)
    print(f"Task B after Task B with EWC: Accuracy: {accuracy_taskB_ewc_val:.4f}, Precision: {precision_taskB_ewc_val:.4f}, Recall: {recall_taskB_ewc_val:.4f}, Kappa: {kappa_taskB_ewc_val:.4f}")
    results_val["B_ewc_val"]["accuracy"].append(accuracy_taskB_ewc_val)
    results_val["B_ewc_val"]["precision"].append(precision_taskB_ewc_val)
    results_val["B_ewc_val"]["recall"].append(recall_taskB_ewc_val)
    results_val["B_ewc_val"]["kappa"].append(kappa_taskB_ewc_val)

    # Evaluate performance on Task A after training on Task B with EWC
    accuracy_taskA_final_ewc, precision_taskA_final_ewc, recall_taskA_final_ewc, kappa_taskA_final_ewc = evaluate(modelA, test_A_loader, criterion)
    print(f"Task A - Final with EWC: Accuracy: {accuracy_taskA_final_ewc}, Precision: {precision_taskA_final_ewc}, Recall: {recall_taskA_final_ewc}, Cohen's Kappa: {kappa_taskA_final_ewc}")
    results["A_after_B_ewc"]["accuracy"].append(accuracy_taskA_final_ewc)
    results["A_after_B_ewc"]["precision"].append(precision_taskA_final_ewc)
    results["A_after_B_ewc"]["recall"].append(recall_taskA_final_ewc)
    results["A_after_B_ewc"]["kappa"].append(kappa_taskA_final_ewc)

    # Evaluate performance on Task A after training on Task B with EWC - Validation
    accuracy_taskA_after_B_ewc_val, precision_taskA_after_B_ewc_val, recall_taskA_after_B_ewc_val, kappa_taskA_after_B_ewc_val = evaluate(modelA, val_A_loader, criterion)
    print(f"Task A after Task B with EWC: Accuracy: {accuracy_taskA_after_B_ewc_val:.4f}, Precision: {precision_taskA_after_B_ewc_val:.4f}, Recall: {recall_taskA_after_B_ewc_val:.4f}, Cohen's Kappa: {kappa_taskA_after_B_ewc_val:.4f}")
    results_val["A_after_B_ewc_val"]["accuracy"].append(accuracy_taskA_after_B_ewc_val)
    results_val["A_after_B_ewc_val"]["precision"].append(precision_taskA_after_B_ewc_val)
    results_val["A_after_B_ewc_val"]["recall"].append(recall_taskA_after_B_ewc_val)
    results_val["A_after_B_ewc_val"]["kappa"].append(kappa_taskA_after_B_ewc_val)

    with open(f'{output_dir}/epoch_stats_til_bench_test_feb25_5.json', 'w') as f:
        json.dump(epoch_stats, f, indent=4)


sys.stdout.flush()


