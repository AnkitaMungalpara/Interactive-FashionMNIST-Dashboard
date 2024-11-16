import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FashionMNISTNet
from flask import Flask, render_template, jsonify
import threading
import random
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Global variables to store training progress
epoch_losses = []
epoch_accuracies = []
epochs = []
status = "Initializing"
prediction_image = None
lock = threading.Lock()

# Fashion MNIST classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_metrics')
def get_metrics():
    with lock:
        return jsonify({
            'losses': epoch_losses,
            'accuracies': epoch_accuracies,
            'epochs': epochs,
            'status': status,
            'prediction_image': prediction_image
        })

def generate_prediction_grid(model, test_loader, device):
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    # Get random samples
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    
    # Get predictions
    with torch.no_grad():
        images = images[:10].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Plot images with predictions
    for idx, (img, pred) in enumerate(zip(images, predicted)):
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img * 0.5 + 0.5  # Denormalize
        axes[idx].imshow(img.squeeze(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'Pred: {classes[pred]}')

    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    return (predicted == targets).sum().item() / targets.size(0)

def train_model():
    global status, epoch_losses, epoch_accuracies, epochs, prediction_image
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Initialize model, loss, and optimizer
    model = FashionMNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    with lock:
        status = "Training"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            acc = calculate_accuracy(output, target)
            running_acc += acc
            running_loss += loss.item()
            total_batches += 1

        # Calculate epoch metrics
        epoch_loss = running_loss / total_batches
        epoch_acc = running_acc / total_batches
        
        with lock:
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)
            epochs.append(epoch + 1)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # Generate prediction grid
    with lock:
        status = "Generating Predictions"
        prediction_image = generate_prediction_grid(model, test_loader, device)
        status = "Training Complete"

    # Save model
    torch.save(model.state_dict(), 'fashion_mnist_model.pth')

def start_server():
    app.run(port=5000)

if __name__ == '__main__':
    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # Start training in the main thread
    train_model() 