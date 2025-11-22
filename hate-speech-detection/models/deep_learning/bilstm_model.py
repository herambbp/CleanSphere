"""
BiLSTM Model for Hate Speech Classification - PyTorch Implementation
Bidirectional Long Short-Term Memory network

Reads text in both directions (forward and backward) for better context understanding.

Architecture:
- Embedding Layer (learns word representations)
- Bidirectional LSTM Layer (captures patterns in both directions)
- Dropout Layers (prevents overfitting)
- Dense Layers (classification)

GPU Accelerated with CUDA support
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import (
    BILSTM_CONFIG, NUM_CLASSES, MODEL_FILES,
    USE_EARLY_STOPPING, USE_MODEL_CHECKPOINT, CLASS_WEIGHTS
)

# ==================== BiLSTM NETWORK ====================

class BiLSTMNet(nn.Module):
    """PyTorch Bidirectional LSTM network architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        num_classes: int,
        dropout: float = 0.5
    ):
        super(BiLSTMNet, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            batch_first=True,
            bidirectional=True,  # Key difference from LSTM
            dropout=dropout if dropout > 0 else 0
        )
        self.dropout1 = nn.Dropout(dropout)
        # Input size is lstm_units * 2 because of bidirectional
        self.fc1 = nn.Linear(lstm_units * 2, 64)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        # Concatenate last hidden states from both directions
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]
        lstm_out = torch.cat((hidden_forward, hidden_backward), dim=1)
        out = self.dropout1(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out


class BiLSTMModel:
    """Bidirectional LSTM-based text classifier for hate speech detection."""
    
    def __init__(self, config: Dict = None):
        self.config = config or BILSTM_CONFIG
        self.model = None
        self.history = None
        self.training_time = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vocab_size = self.config['vocab_size']
        self.embedding_dim = self.config['embedding_dim']
        self.lstm_units = self.config['lstm_units']
        self.dropout = self.config['dropout']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.learning_rate = self.config['learning_rate']
        
        print(f"[BiLSTM] Using device: {self.device}")
    
    def build_model(self):
        """Build BiLSTM model architecture."""
        self.model = BiLSTMNet(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_classes=NUM_CLASSES,
            dropout=self.dropout
        ).to(self.device)
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """Train the BiLSTM model."""
        if self.model is None:
            self.build_model()
        
        X_train_tensor = torch.LongTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.LongTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            val_loader = None
        
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(list(CLASS_WEIGHTS.values())).to(self.device)
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_val_acc = 0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = train_correct / train_total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_epoch_loss = val_loss / len(val_loader)
                val_epoch_acc = val_correct / val_total
                history['val_loss'].append(val_epoch_loss)
                history['val_accuracy'].append(val_epoch_acc)
                
                if USE_EARLY_STOPPING:
                    if val_epoch_acc > best_val_acc:
                        best_val_acc = val_epoch_acc
                        patience_counter = 0
                        if USE_MODEL_CHECKPOINT:
                            self.save()
                    else:
                        patience_counter += 1
                        if patience_counter >= 3:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
                
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
                          f"val_loss: {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")
        
        self.training_time = time.time() - start_time
        self.history = history
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch_X, in loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        self.model.eval()
        X_tensor = torch.LongTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        probabilities = []
        with torch.no_grad():
            for batch_X, in loader:
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, verbose: int = 0) -> Dict:
        """Evaluate model on test data."""
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before evaluation")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if verbose > 0:
            from config import CLASS_LABELS
            target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
            report = classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0)
            print("\nClassification Report:")
            print(report)
        
        return metrics
    
    def save(self, filepath: str = None):
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        filepath = filepath or str(MODEL_FILES['bilstm']).replace('.keras', '.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
            'training_time': self.training_time
        }, filepath)
    
    @staticmethod
    def load(filepath: str = None) -> 'BiLSTMModel':
        """Load model from disk."""
        filepath = filepath or str(MODEL_FILES['bilstm']).replace('.keras', '.pt')
        checkpoint = torch.load(filepath, map_location='cpu')
        
        bilstm_model = BiLSTMModel(config=checkpoint['config'])
        bilstm_model.build_model()
        bilstm_model.model.load_state_dict(checkpoint['model_state_dict'])
        bilstm_model.history = checkpoint.get('history')
        bilstm_model.training_time = checkpoint.get('training_time')
        
        return bilstm_model
    
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
        else:
            print(self.model)
            print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            return {'model_type': 'BiLSTM', 'built': False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'BiLSTM',
            'built': True,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'total_lstm_output': self.lstm_units * 2,
            'max_length': self.max_length,
            'num_classes': NUM_CLASSES,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': str(self.device)
        }
        
        if self.training_time is not None:
            info['training_time'] = self.training_time
        if self.history is not None:
            info['final_train_accuracy'] = self.history['accuracy'][-1]
            if 'val_accuracy' in self.history and self.history['val_accuracy']:
                info['final_val_accuracy'] = self.history['val_accuracy'][-1]
        
        return info
    
    def __repr__(self):
        if self.model is None:
            return "BiLSTMModel(not built)"
        else:
            params = sum(p.numel() for p in self.model.parameters())
            return f"BiLSTMModel(params={params:,}, lstm_units={self.lstm_units}x2, device={self.device})"


def create_bilstm_model(config: Dict = None) -> BiLSTMModel:
    """Create and build BiLSTM model."""
    model = BiLSTMModel(config=config)
    model.build_model()
    return model