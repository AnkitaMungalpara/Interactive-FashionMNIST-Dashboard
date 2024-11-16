import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FashionMNISTNet
from flask import Flask, render_template, jsonify, request
import threading
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from queue import Queue

app = Flask(__name__)

@dataclass
class ModelConfig:
    layer_sizes: List[int]
    optimizer_type: str = 'Adam'
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001

class TrainingManager:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        """Reset all state variables"""
        self.models_metrics = {}
        self.status = "Ready"
        self.prediction_images = {}
        self.lock = threading.Lock()
        self.active_threads = []

    def start_training(self, configs):
        """Start training multiple models in parallel"""
        # Reset state before starting new training
        self.reset_state()
        
        model_ids = []
        
        # Initialize metrics for all models
        for i in range(len(configs)):
            model_id = str(i)
            model_ids.append(model_id)
            with self.lock:
                self.models_metrics[model_id] = {
                    'losses': [],
                    'accuracies': [],
                    'epochs': [],
                    'config': configs[i],
                    'training_complete': False
                }

        # Start a separate thread for each model
        for i, config in enumerate(configs):
            thread = threading.Thread(
                target=self.train_model, 
                args=(str(i), config)
            )
            thread.daemon = True
            self.active_threads.append(thread)
            thread.start()

        return model_ids

    def train_model(self, model_id: str, config: ModelConfig):
        """Train a single model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with self.lock:
            self.status = f"Initializing Model {model_id}"

        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=2  # Add workers for faster data loading
        )
        
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        # Model initialization
        model = FashionMNISTNet(layer_sizes=config.layer_sizes).to(device)
        criterion = nn.CrossEntropyLoss()
        
        if config.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

        # Training loop
        total_epochs = config.epochs
        for epoch in range(total_epochs):
            with self.lock:
                self.status = f"Training Models... Model {model_id}: Epoch {epoch + 1}/{total_epochs}"
            
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            total_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                acc = calculate_accuracy(output, target)
                running_acc += acc
                running_loss += loss.item()
                total_batches += 1

                # Update metrics every few batches for smoother plotting
                if batch_idx % 5 == 0:
                    current_loss = running_loss / (batch_idx + 1)
                    current_acc = running_acc / (batch_idx + 1)
                    current_epoch = epoch + batch_idx/len(train_loader)
                    
                    with self.lock:
                        self.models_metrics[model_id]['losses'].append(current_loss)
                        self.models_metrics[model_id]['accuracies'].append(current_acc)
                        self.models_metrics[model_id]['epochs'].append(current_epoch)

        # Generate prediction grid after training
        with self.lock:
            self.status = f"Generating Predictions for Model {model_id}"
            self.prediction_images[model_id] = generate_prediction_grid(model, test_loader, device)
            self.models_metrics[model_id]['training_complete'] = True
            
            # Check if all models are complete
            all_complete = all(
                metrics.get('training_complete', False) 
                for metrics in self.models_metrics.values()
            )
            if all_complete:
                self.status = "Training Complete"

training_manager = TrainingManager()

# Fashion MNIST classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_models', methods=['POST'])
def train_models():
    data = request.get_json()
    
    configs = [
        ModelConfig(
            layer_sizes=data['model1_layers'],
            optimizer_type=data['model1_optimizer'],
            batch_size=int(data['model1_batch_size']),
            epochs=int(data['model1_epochs']),
            learning_rate=float(data['model1_lr'])
        ),
        ModelConfig(
            layer_sizes=data['model2_layers'],
            optimizer_type=data['model2_optimizer'],
            batch_size=int(data['model2_batch_size']),
            epochs=int(data['model2_epochs']),
            learning_rate=float(data['model2_lr'])
        )
    ]
    
    model_ids = training_manager.start_training(configs)
    return jsonify({'model_ids': model_ids})

@app.route('/get_metrics')
def get_metrics():
    with training_manager.lock:
        return jsonify({
            'models_metrics': training_manager.models_metrics,
            'status': training_manager.status,
            'prediction_images': training_manager.prediction_images
        })

def generate_prediction_grid(model, test_loader, device):
    model.eval()
    plt.ioff()  # Turn off interactive mode
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
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)  # Explicitly close the figure
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    return (predicted == targets).sum().item() / targets.size(0)

def start_server():
    app.run(port=5000)

if __name__ == '__main__':
    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()