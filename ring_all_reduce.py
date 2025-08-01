import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, Subset
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import matplotlib.pyplot as plt
import time
from multiprocessing import Queue


############################################################# Data Transformation / Preprocessing ###############################################

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

# Select 100 random indices from the training dataset
indices = torch.randperm(len(train_dataSet))[:100]

# Create a subset of the original dataset
train_data_subset = Subset(train_dataSet, indices)

# Now split into 4 subsets of 25 examples each
subset_size = 25
train_subsets = random_split(train_data_subset, [subset_size] * 4)


# Computing the mean manually
def compute_mean(list_of_lists):
    transposed = zip(*list_of_lists)   # Transpose to group values by epoch
    return [sum(group) / len(group) for group in transposed]





############################################################## LeNet-5 Model #############################################################

class LeNet_5(nn.Module):
    # Initialization 
    def __init__(self):
        super(LeNet_5, self).__init__()                  
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5)     # First Convolutional Layer: input channel = 1, output filters = 6                             
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)# Average Pooling 
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5)    # Second Convolutional Layer: input channel = 6, output filters = 16
        self.conv_3 = nn.Conv2d(16, 120, kernel_size=5)  # Third Convolutional Layer: input channel = 16, output filters = 120
        self.act = nn.Tanh()                             # Activation function      
        self.fc_1 = nn.Linear(120, 84)                   # Fully connected layers: input neurons(120) → output neurons(84)
        self.fc_2 = nn.Linear(84, 10)                    # Fully connected output layers: input neurons(120) → output classes(10)


    # Forward pass of Network
    def forward(self, input_tensor):
        input_tensor = self.act(self.conv_1(input_tensor))  # Apply activation after first convolution  
        input_tensor = self.pool(input_tensor)              # Perform pooling operation  
        input_tensor = self.act(self.conv_2(input_tensor))  # Apply activation after second convolution  
        input_tensor = self.pool(input_tensor)              # Perform pooling operation again  
        input_tensor = self.act(self.conv_3(input_tensor))  # Apply activation after third convolution  
        input_tensor = input_tensor.view(-1, 120)           # Reshape tensor before passing to Fully connected layer  
        input_tensor = self.act(self.fc_1(input_tensor))    # Apply activation after first Fully connected layer  
        input_tensor = self.fc_2(input_tensor)              # Final Fully connected layer / output layer 
        return input_tensor 

  
  
  

############################################################# Training Function with Ring All-Reduce ############################################################

def train(rank, world_size, model, train_subsets, num_epochs=100, queue=None):
    # Initialize the distributed backend
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initializing the distributed backend
    dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=world_size)

    # Moving model to the current device
    device = torch.device("cpu")
    model = torch.nn.parallel.DistributedDataParallel(model) # Using DistributedDataParallel for Data Parallelism

    # Get the subset for this process
    train_subset = train_subsets[rank]
    
    # Extract dataset and indices from above subset
    dataset = train_subset.dataset
    indices = train_subset.indices
    
    # Create a new subset for the DataLoader
    subset_dataset = Subset(dataset, indices)
    
    # Create a DataLoader for the subset
    train_loader = DataLoader(subset_dataset, batch_size=25, shuffle=True)  

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Increased learning rate

    # Lists to store metrics(accuracy and loss)
    train_losses = []
    train_accuracies = []

    # Start time of training
    start_time = time.time()

    # Running for each of the randomly selected data
    for epoch in range(num_epochs):
        model.train()       # Set model to training mode
        epoch_loss = 0.0    # Initialize epoch loss
        correct = 0         # Initialize count of correct predictions 
        total = 0           # Initialize total sample count

        # Iterate over training data
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device) # Move data to the appropriate device (here CPU)

            # Forward pass: compute predictions
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Compute loss

            # Backward pass: compute gradients
            optimizer.zero_grad()   # Reset gradients to avoid accumulation
            loss.backward()         # Compute gradients

            # Ring All-Reduce implementation
            for param in model.parameters():
                # Send gradients to the next process in the ring
                next_rank = (rank + 1) % world_size
                prev_rank = (rank - 1) % world_size

                # Non-blocking send and receive
                send_req = dist.isend(param.grad.data, dst=next_rank)
                recv_buffer = torch.zeros_like(param.grad.data)
                recv_req = dist.irecv(recv_buffer, src=prev_rank)

                # Wait for send and receive to complete
                send_req.wait()
                recv_req.wait()

                # Accumulate and average the gradients
                param.grad.data += recv_buffer
                param.grad.data /= world_size

            # Update model parameters
            optimizer.step()

            # Compute training accuracy
            _, predicted = outputs.max(1)                   # Predicted class index
            total += targets.size(0)                        # Update total sample count
            correct += predicted.eq(targets).sum().item()   # Counting correct predictions

            # Loss for tracking epoch 
            epoch_loss += loss.item()

        # Computing average loss 
        epoch_loss /= len(train_loader)

        # Computing accuracy percentage
        epoch_accuracy = 100.0 * correct / total

        # Store training loss & accuracy
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Print metrics
        print(f"Rank {rank}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Cleanup distributed training process
    dist.destroy_process_group()
    
    # End time for total training
    end_time = time.time()
    training_time = end_time - start_time  # Total training time

    # Printing training duration for analysis
    print(f"Rank {rank}, Training Time: {training_time:.2f} seconds")

    # Sending metrics (loss, accuracy, training time) to the main process
    if queue is not None:
        queue.put((train_losses, train_accuracies, training_time))





######################################################## Main Function ########################################################

def main():
    # Initialization of model
    model = LeNet_5()
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)

    # Create a queue to collect metrics
    queue = Queue()

    # Launch 4 processes for data parallelism, each of which will run on a different core
    world_size = 4
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train, args=(rank, world_size, model, train_subsets, 100, queue))  # Create a new process for distributed training
        p.start()
        processes.append(p)

    # Waiting for all processes to get finish
    for p in processes:
        p.join()

    # Collecting metrics from queue
    all_train_losses = []
    all_train_accuracies = []
    all_training_times = []
    while not queue.empty():
        losses, accuracies, training_time = queue.get()
        all_train_losses.append(losses)
        all_train_accuracies.append(accuracies)
        all_training_times.append(training_time)

                        ##############  Average metrics across processes  #############

    # Calculating mean 
    avg_train_losses = compute_mean(all_train_losses)
    avg_train_accuracies = compute_mean(all_train_accuracies)
    avg_training_time = sum(all_training_times) / len(all_training_times)

    # Print averaged metrics
    print("Averaged Train Losses:", avg_train_losses)
    print("Averaged Train Accuracies:", avg_train_accuracies)
    print(f"Average Training Time: {avg_training_time:.2f} seconds")

    # Plotting training loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()

    # Plotting training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print final accuracy
    print(f"Final Training Accuracy: {avg_train_accuracies[-1]:.2f}%")


if __name__ == "__main__":
    main()