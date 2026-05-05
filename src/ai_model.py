"""
Deep learning models for TRM forecasting
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AIModels:
    """Deep learning models for time series forecasting"""
    
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.models = {}
        self.history = {}
        self.model_path = Path("models/ai")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def build_lstm_model(self, units: list = [64, 32], dropout_rate: float = 0.2):
        """Build LSTM model"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae', 'mape'])
        
        self.models['lstm'] = model
        return model
    
    def build_bilstm_model(self, units: list = [64, 32], dropout_rate: float = 0.2):
        """Build Bidirectional LSTM model"""
        model = Sequential()
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units[0], return_sequences=True), 
                               input_shape=self.input_shape))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae', 'mape'])
        
        self.models['bilstm'] = model
        return model
    
    def build_gru_model(self, units: list = [64, 32], dropout_rate: float = 0.2):
        """Build GRU model"""
        model = Sequential()
        
        model.add(GRU(units[0], return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(dropout_rate))
        
        model.add(GRU(units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae', 'mape'])
        
        self.models['gru'] = model
        return model
    
    def build_attention_lstm(self, units: int = 64):
        """Build LSTM with attention mechanism"""
        from tensorflow.keras.layers import Layer, Concatenate
        from tensorflow.keras import backend as K
        
        class AttentionLayer(Layer):
            def __init__(self, **kwargs):
                super(AttentionLayer, self).__init__(**kwargs)
            
            def build(self, input_shape):
                self.W = self.add_weight(name='attention_weight', 
                                       shape=(input_shape[-1], 1),
                                       initializer='random_normal',
                                       trainable=True)
                self.b = self.add_weight(name='attention_bias',
                                       shape=(input_shape[1], 1),
                                       initializer='zeros',
                                       trainable=True)
                super(AttentionLayer, self).build(input_shape)
            
            def call(self, x):
                e = K.tanh(K.dot(x, self.W) + self.b)
                e = K.squeeze(e, axis=-1)
                alpha = K.softmax(e)
                alpha = K.expand_dims(alpha, axis=-1)
                context = x * alpha
                context = K.sum(context, axis=1)
                return context
        
        # Build model with attention
        inputs = tf.keras.Input(shape=self.input_shape)
        lstm_out = LSTM(units, return_sequences=True)(inputs)
        attention_out = AttentionLayer()(lstm_out)
        dense1 = Dense(16, activation='relu')(attention_out)
        dropout = Dropout(0.2)(dense1)
        outputs = Dense(1)(dropout)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae', 'mape'])
        
        self.models['attention_lstm'] = model
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, 
                   batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train a specific model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(self.model_path / f'{model_name}_best.h5', 
                          monitor='val_loss', save_best_only=True)
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history[model_name] = history
        return history
    
    def predict(self, model_name: str, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions from a trained model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        predictions = model.predict(X_test)
        return predictions.flatten()
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                       y_pred: np.ndarray) -> dict:
        """Evaluate model performance"""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.models.items():
            model.save(self.model_path / f"{name}.h5")
        logger.info("All AI models saved")
    
    def load_models(self):
        """Load saved models"""
        for model_file in self.model_path.glob("*.h5"):
            name = model_file.stem
            self.models[name] = load_model(model_file)
        logger.info(f"Loaded {len(self.models)} AI models")

if __name__ == "__main__":
    # Test AI models
    from preprocess import TRMPreprocessor
    import pandas as pd
    
    df = pd.read_csv("data/processed/trm_historical.csv")
    preprocessor = TRMPreprocessor()
    df_features = preprocessor.create_features(df)
    X, y = preprocessor.prepare_sequence_data(df_features)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train models
    ai_models = AIModels(input_shape=(X.shape[1], X.shape[2]))
    ai_models.build_lstm_model()
    ai_models.train_model('lstm', X_train, y_train, X_test, y_test, epochs=50)
    
    predictions = ai_models.predict('lstm', X_test)
    metrics = ai_models.evaluate_model('lstm', y_test, predictions)
    print(f"LSTM metrics: {metrics}")