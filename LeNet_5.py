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



########################################## LeNet-5 Model ##########################################

class LeNet_5(nn.Module):
    # Initialization 
    def __init__(self):
        super(LeNet_5, self).__init__()                  
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5)     # First Convolutional Layer: input channel = 1, output filters = 6                             #
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)# Average Pooling 
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)     # Second Convolutional Layer: input channel = 6, output filters = 16
        self.conv_3 = nn.Conv2d(16, 120, kernel_size=5)   # Third Convolutional Layer: input channel = 16, output filters = 120
        self.act = nn.Tanh()                             # Activation function
        
        self.fc_1 = nn.Linear(120, 84)                    # Fully connected layers: input neurons(120) → output neurons(84)
        self.fc_2 = nn.Linear(84, 10)                     # Fully connected output layers: input neurons(120) → output classes(10)


    #  Forward pass of Network
    def forward(self, input_tensor):
        input_tensor = self.act(self.conv_1(input_tensor))  # Apply activation after first convolution  
        input_tensor = self.pool(input_tensor)             # Perform pooling operation  
        input_tensor = self.act(self.conv_2(input_tensor))  # Apply activation after second convolution  
        input_tensor = self.pool(input_tensor)             # Perform pooling operation again  
        input_tensor = self.act(self.conv_3(input_tensor))  # Apply activation after third convolution  
        input_tensor = input_tensor.view(-1, 120)          # Reshape tensor before passing to Fully connected layer  
        input_tensor = self.act(self.fc_1(input_tensor))    # Apply activation after first Fully connected layer  
        input_tensor = self.fc_2(input_tensor)              # Final Fully connected layer / output layer 
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
    plt.plot(epochs_range, test_losses, color='red', linestyle='--', linewidth=2, label='Testing Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Testing Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, color='green', linestyle='-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs_range, test_accs, color='orange', linestyle='--', linewidth=2, label='Testing Accuracy')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device (GPU/CPU)  
    model = LeNet_5().to(device)  # Initialize model and move it to the selected device  
    loss_fn = nn.CrossEntropyLoss()  # Define loss function  
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Use Adam optimizer  

    patience = 5            # Max epochs to wait without improvement before stopping  
    min_delta = 0.01        # Minimum accuracy improvement required to reset patience  
    best_accuracy = 0.0     # Track the best test accuracy achieved  
    no_improve_epochs = 0   # Counter for epochs without significant improvement  

    # Lists to store loss and accuracy values  
    train_losses, train_accuracy = [], []  
    test_losses, test_accuracy = [], []  

    epoch = 0  
    while True:  # Infinite loop until early stopping condition is met  
        epoch += 1  
        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_fn)  
        test_loss, test_acc = evaluate(model, device, test_loader, loss_fn)  

        # Store training and evaluation metrics  
        train_losses.append(train_loss)  
        train_accuracy.append(train_acc)  
        test_losses.append(test_loss)  
        test_accuracy.append(test_acc)  

        # Display progress  
        print(f"Epoch [{epoch}] "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")      
         

        # Check if test accuracy has significantly improved  
        if test_acc > best_accuracy + min_delta:  
            best_accuracy = test_acc    # Update best accuracy  
            no_improve_epochs = 0       # Reset counter  
        else:  
            no_improve_epochs += 1  # Increment counter  

        # Stop training if there is no improvement for 'patience' consecutive epochs  
        if no_improve_epochs >= patience:    
            break

    plot_metrics(epoch, train_losses, test_losses, train_accuracy, test_accuracy)


if __name__ == "__main__":
    main()
      






    
    

 