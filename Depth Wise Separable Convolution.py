import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


########################################## Data Transformation / Preprocessing ##########################################

# Transforming data:
# Padding 28x28 to 32x32 to resize image
# Convert to tensor for suitable format
# Normalizing scale for training
transform_data = transforms.Compose([
    transforms.Pad(2),                   # pad 28x28 to 32x32
    transforms.ToTensor(),               # converting to tensor 
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1,1]
])


# Loading MNIST dataset
train_dataSet = MNIST(root='./data', train=True, download=True, transform=transform_data)
test_dataSet  = MNIST(root='./data', train=False, download=True, transform=transform_data)

# Creating data loaders for batching and shuffing
train_loader = DataLoader(train_dataSet, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataSet, batch_size=64, shuffle=False)



########################################## Depth-wise Separable Convolution  ##########################################

class LeNet_5_Depth_Wise(nn.Module):
    def __init__(self):
        super(LeNet_5_Depth_Wise, self).__init__()
        
        # Depth-wise separable convolution layers
        self.conv1_depthwise = nn.Conv2d(1, 1, kernel_size=5, groups=1)  # Depthwise
        self.conv1_pointwise = nn.Conv2d(1, 6, kernel_size=1)  # Pointwise
        
        self.conv2_depthwise = nn.Conv2d(6, 6, kernel_size=5, groups=6)  # Depthwise
        self.conv2_pointwise = nn.Conv2d(6, 16, kernel_size=1)  # Pointwise
        
        self.conv3_depthwise = nn.Conv2d(16, 16, kernel_size=5, groups=16)  # Depthwise
        self.conv3_pointwise = nn.Conv2d(16, 120, kernel_size=1)  # Pointwise
        
        # Activation function
        self.act = nn.Tanh()
        
        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, input_tensor):
        # First convolutional block
        input_tensor = self.act(self.conv1_depthwise(input_tensor))  # Apply depthwise convolution
        input_tensor = self.conv1_pointwise(input_tensor)  # Apply pointwise convolution
        input_tensor = self.pool(input_tensor)  # Apply pooling
        
        # Second convolutional block
        input_tensor = self.act(self.conv2_depthwise(input_tensor))
        input_tensor = self.conv2_pointwise(input_tensor)
        input_tensor = self.pool(input_tensor)
        
        # Third convolutional block
        input_tensor = self.act(self.conv3_depthwise(input_tensor))
        input_tensor = self.conv3_pointwise(input_tensor)
        
        # Flatten for fully connected layer
        input_tensor = input_tensor.view(-1, 120)
        
        # Fully connected layers with activation
        input_tensor = self.act(self.fc1(input_tensor))
        input_tensor = self.fc2(input_tensor)  # Output layer
        
        return input_tensor 
    
########################################## Training ##########################################
   
 
def train(model, device, data_train_loader, optimizer, loss_func):
    model.train()  # Set model to training mode  
    total_loss = 0.0  
    correct_predictions = 0  
    total_samples = 0  

    for inputs, targets in data_train_loader:  
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device  

        optimizer.zero_grad()                   # Reset gradients  
        predictions = model(inputs)             # Forward pass  
        loss = loss_func(predictions, targets)  # Compute loss  
        loss.backward()                         # Backpropagation  
        optimizer.step()                        # Update model parameters  

        total_loss = total_loss + (loss.item() * inputs.size(0))  # Accumulate loss  
        _, predicted_labels = predictions.max(1)                  # Get predicted class labels  
        total_samples = total_samples + targets.size(0)           # Count total samples  
        correct_predictions = correct_predictions + (predicted_labels.eq(targets).sum().item() ) # Count correct predictions  

    epoch_loss = total_loss / total_samples                       # Compute average loss  
    epoch_accuracy = 100.0 * correct_predictions / total_samples  # Calculate accuracy  
    return epoch_loss, epoch_accuracy                             # Return loss and accuracy for the epoch  



########################################## Evaluation ##########################################


def evaluate(model, device, data_test_loader, loss_fn):
    model.eval()  # Set model to evaluation mode  
    total_loss = 0.0  
    correct_predictions = 0  
    total_samples = 0  

    with torch.no_grad():  # Disable gradient computation for evaluation  
        for inputs, targets in data_test_loader:  
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the specified device  

            outputs = model(inputs)                     # Perform forward pass  
            loss = loss_fn(outputs, targets)            # Compute loss  
            total_loss += loss.item() * inputs.size(0)  # Accumulate total loss  

            _, predicted_labels = outputs.max(1)        # Get the predicted class labels  
            total_samples += targets.size(0)            # Count total samples  
            correct_predictions += predicted_labels.eq(targets).sum().item()  # Count correct predictions  

    epoch_loss = total_loss / total_samples                       # Compute average loss  
    epoch_accuracy = 100.0 * correct_predictions / total_samples  # Calculate accuracy  

    return epoch_loss, epoch_accuracy  # Return the loss and accuracy values  



################################## Plotting loss and accuracy curves versus epochs ################################


def plot_metrics(epoch, train_losses, test_losses, train_accs, test_accs):
    epochs_range = range(1, epoch + 1)
    
    # Create a figure with two subplots
    plt.figure(figsize=(14, 6))
    
    # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, color='blue', linestyle='-', linewidth=2, label='Training Loss')
    plt.plot(epochs_range, test_losses, color='red', linestyle='--', linewidth=2, label='Validation Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, color='green', linestyle='-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs_range, test_accs, color='orange', linestyle='--', linewidth=2, label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Testing Accuracy', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()



########################################## Main Function ##########################################

def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model, loss function, and optimizer
    network = LeNet_5_Depth_Wise().to(device)
    loss_fn = nn.CrossEntropyLoss()
    model_optimizer = optim.SGD(network.parameters(), lr=0.01)
    
    # Early stopping parameters
    early_stop_patience = 5       # Number of epochs to wait for improvement
    min_improvement = 0.01        # Minimum improvement in validation accuracy
    best_test_accuracy = 0.0
    no_improvement_count = 0      # Counter for epochs without improvement
    
    # Lists to store training and validation metrics
    train_losses, train_accuracy= [], []
    test_losses, test_accuracy = [], []
    
    # Start training loop
    epoch = 0
    while True:
        epoch += 1
        
        # Training phase
        train_loss, train_acc = train(network, device, train_loader, model_optimizer, loss_fn)
        
        # Testing phase
        test_loss, test_acc = evaluate(network, device, test_loader, loss_fn)
        
        # Store metrics for plotting
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        
        # Print epoch results
        print(f"Epoch [{epoch}] "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Check for improvement in testing accuracy
        if test_acc > (best_test_accuracy + min_improvement):
            best_test_accuracy = test_acc
            no_improvement_count = 0  # Reset counter
        else:
            no_improvement_count += 1  # Increment counter
        
        # Early stopping condition
        if no_improvement_count >= early_stop_patience:
            break
        
    plot_metrics(epoch, train_losses, test_losses, train_accuracy, test_accuracy)




if __name__ == "__main__":
    main() 
