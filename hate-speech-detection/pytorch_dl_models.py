"""
PyTorch Deep Learning Models with CUDA Support
LSTM, BiLSTM, CNN models optimized for GPU training

This replaces TensorFlow/Keras models with PyTorch equivalents
maintaining same architecture and parameters for comparable accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[PYTORCH] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[CUDA] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[CUDA] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ==================== DATA HANDLING ====================

class TextDataset(Dataset):
    """PyTorch Dataset for text sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: Padded sequences (N, max_len)
            labels: Labels (N,)
        """
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_data_loader(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader"""
    dataset = TextDataset(sequences, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )


# ==================== LSTM MODEL ====================

class LSTMClassifier(nn.Module):
    """
    LSTM model for text classification
    Architecture matches Keras LSTM model
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        lstm_units: int = 64,
        num_classes: int = 3,
        max_length: int = 100,
        dropout: float = 0.5
    ):
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # No dropout for single layer
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense layer
        self.fc = nn.Linear(lstm_units, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input sequences (batch_size, max_length)
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, max_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Take last output
        last_output = lstm_out[:, -1, :]  # (batch, lstm_units)
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # Dense
        output = self.fc(dropped)  # (batch, num_classes)
        
        return output
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'model_type': 'lstm'
        }


# ==================== BiLSTM MODEL ====================

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM model for text classification
    Architecture matches Keras BiLSTM model
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        lstm_units: int = 64,
        num_classes: int = 3,
        max_length: int = 100,
        dropout: float = 0.5
    ):
        super(BiLSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense layer (input is 2*lstm_units due to bidirectional)
        self.fc = nn.Linear(lstm_units * 2, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input sequences (batch_size, max_length)
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, max_len, embed_dim)
        
        # BiLSTM
        bilstm_out, (hidden, cell) = self.bilstm(embedded)
        
        # Take last output
        last_output = bilstm_out[:, -1, :]  # (batch, lstm_units*2)
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # Dense
        output = self.fc(dropped)  # (batch, num_classes)
        
        return output
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'model_type': 'bilstm'
        }


# ==================== CNN MODEL ====================

class CNNClassifier(nn.Module):
    """
    CNN model for text classification
    Architecture matches Keras CNN model with multiple filter sizes
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        embedding_dim: int = 128,
        num_filters: int = 128,
        filter_sizes: List[int] = [3, 4, 5],
        num_classes: int = 3,
        max_length: int = 100,
        dropout: float = 0.5
    ):
        super(CNNClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Convolutional layers for different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense layer
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input sequences (batch_size, max_length)
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, max_len, embed_dim)
        
        # Transpose for Conv1d: (batch, embed_dim, max_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch, num_filters, L)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)
        
        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        
        # Dropout
        dropped = self.dropout(concatenated)
        
        # Dense
        output = self.fc(dropped)  # (batch, num_classes)
        
        return output
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'filter_sizes': self.filter_sizes,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'model_type': 'cnn'
        }


# ==================== TRAINER ====================

class PyTorchDLTrainer:
    """
    PyTorch Deep Learning Trainer with CUDA support
    Trains LSTM, BiLSTM, and CNN models
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        max_length: int = 100,
        num_classes: int = 3
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_classes = num_classes
        self.device = DEVICE
        
        self.models = {}
        self.histories = {}
        self.best_model_name = None
        self.best_accuracy = 0.0
    
    def create_model(
        self,
        model_type: str,
        **kwargs
    ) -> nn.Module:
        """
        Create a model
        
        Args:
            model_type: 'lstm', 'bilstm', or 'cnn'
            **kwargs: Model-specific parameters
        
        Returns:
            PyTorch model
        """
        model_classes = {
            'lstm': LSTMClassifier,
            'bilstm': BiLSTMClassifier,
            'cnn': CNNClassifier
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_classes[model_type](
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            max_length=self.max_length,
            **kwargs
        )
        
        model = model.to(self.device)
        
        print(f"\n[MODEL] Created {model_type.upper()} model")
        print(f"[PARAMS] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[DEVICE] Model on: {next(model.parameters()).device}")
        
        return model
    
    def train_model(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        learning_rate: float = 0.001,
        patience: int = 3,
        verbose: bool = True
    ) -> Dict:
        """
        Train a PyTorch model
        
        Args:
            model: PyTorch model
            model_name: Name for tracking
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            Training history dict
        """
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            if verbose:
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            else:
                train_pbar = train_loader
            
            for sequences, labels in train_pbar:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if verbose and isinstance(train_pbar, tqdm):
                    train_pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{100.*train_correct/train_total:.2f}%"
                    })
            
            # Calculate training metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * sequences.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            # Save history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f}")
                print(f"  Val Loss:   {epoch_val_loss:.4f} - Val Acc:   {epoch_val_acc:.4f}")
            
            # Early stopping
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                if verbose:
                    print(f"  [BEST] New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if verbose:
                    print(f"  [PATIENCE] {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\n[EARLY STOP] Stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n[RESTORED] Best model with val_acc: {best_val_acc:.4f}")
        
        return history
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model on test set
        
        Returns:
            test_loss, test_accuracy, y_true, y_pred
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in tqdm(test_loader, desc="Testing", disable=not verbose):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss = test_loss / len(all_labels)
        test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        if verbose:
            print(f"\n[TEST RESULTS]")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc, np.array(all_labels), np.array(all_preds)
    
    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        save_dir: Path,
        history: Dict = None
    ):
        """Save PyTorch model"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = save_dir / f"{model_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config(),
            'history': history
        }, model_path)
        
        # Save config separately
        config_path = save_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(model.get_config(), f, indent=2)
        
        print(f"[SAVED] Model: {model_path}")
        print(f"[SAVED] Config: {config_path}")
    
    def load_model(
        self,
        model_name: str,
        load_dir: Path
    ) -> nn.Module:
        """Load PyTorch model"""
        load_dir = Path(load_dir)
        model_path = load_dir / f"{model_name}.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']
        
        # Create model
        model = self.create_model(config['model_type'], **config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"[LOADED] Model from: {model_path}")
        
        return model
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_dir: Path = Path('saved_models/pytorch_dl'),
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 0.001
    ):
        """
        Train all models (LSTM, BiLSTM, CNN)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            save_dir: Directory to save models
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
        """
        print(f"\n{'='*80}")
        print("PYTORCH DEEP LEARNING TRAINING WITH CUDA")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(y_train):,}")
        print(f"Val samples: {len(y_val):,}")
        print(f"Test samples: {len(y_test):,}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*80}\n")
        
        # Create data loaders
        train_loader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = create_data_loader(X_val, y_val, batch_size, shuffle=False)
        test_loader = create_data_loader(X_test, y_test, batch_size, shuffle=False)
        
        # Model configurations
        models_config = {
            'lstm': {
                'embedding_dim': 128,
                'lstm_units': 64,
                'dropout': 0.5
            },
            'bilstm': {
                'embedding_dim': 128,
                'lstm_units': 64,
                'dropout': 0.5
            },
            'cnn': {
                'embedding_dim': 128,
                'num_filters': 128,
                'filter_sizes': [3, 4, 5],
                'dropout': 0.5
            }
        }
        
        # Train each model
        for model_name, config in models_config.items():
            print(f"\n{'#'*80}")
            print(f"# TRAINING {model_name.upper()}")
            print(f"{'#'*80}\n")
            
            # Create model
            model = self.create_model(model_name, **config)
            
            # Train
            history = self.train_model(
                model=model,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                patience=3,
                verbose=True
            )
            
            # Evaluate
            test_loss, test_acc, y_true, y_pred = self.evaluate_model(
                model=model,
                test_loader=test_loader,
                verbose=True
            )
            
            # Save
            self.models[model_name] = model
            self.histories[model_name] = history
            self.save_model(model, model_name, save_dir, history)
            
            # Track best model
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_model_name = model_name
            
            # Print summary
            print(f"\n[{model_name.upper()} SUMMARY]")
            print(f"  Best Val Acc: {max(history['val_acc']):.4f}")
            print(f"  Test Acc: {test_acc:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final summary
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best Model: {self.best_model_name.upper()}")
        print(f"Best Test Accuracy: {self.best_accuracy:.4f}")
        print(f"Models saved to: {save_dir}")
        print(f"{'='*80}\n")
        
        return self.models, self.histories


# ==================== INFERENCE ====================

class PyTorchPredictor:
    """
    Predictor for PyTorch models
    """
    
    def __init__(self, model: nn.Module, device: torch.device = DEVICE):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict(self, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions
        
        Args:
            sequences: Padded sequences (N, max_len)
            batch_size: Batch size for prediction
        
        Returns:
            Predicted class indices
        """
        dataset = TextDataset(sequences, np.zeros(len(sequences)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        
        with torch.no_grad():
            for sequences_batch, _ in loader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
        
        return np.array(all_preds)
    
    def predict_proba(self, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            sequences: Padded sequences (N, max_len)
            batch_size: Batch size
        
        Returns:
            Class probabilities (N, num_classes)
        """
        dataset = TextDataset(sequences, np.zeros(len(sequences)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for sequences_batch, _ in loader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)


# ==================== UTILITY FUNCTIONS ====================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Print model summary"""
    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n[INFO] PyTorch Deep Learning Models Module")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Device: {DEVICE}")
    
    # Test model creation
    print("\n[TEST] Creating test models...")
    
    lstm = LSTMClassifier()
    print(f"LSTM parameters: {count_parameters(lstm):,}")
    
    bilstm = BiLSTMClassifier()
    print(f"BiLSTM parameters: {count_parameters(bilstm):,}")
    
    cnn = CNNClassifier()
    print(f"CNN parameters: {count_parameters(cnn):,}")
    
    print("\n[OK] Module ready for training!")