import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings


class RobustStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        X_scaled = X_centered / self.scale_
        # Clip to prevent extreme values
        return np.clip(X_scaled, -5, 5)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_


class ESNParameterPredictor(nn.Module):
    def __init__(self):
        super(ESNParameterPredictor, self).__init__()

        # Define parameter ranges
        self.res_size_range = (10, 1000)  # Reasonable range for reservoir size
        self.k_range = (2, 50)  # Range for k
        self.m_range = (2, 50)  # Range for m

        # Network architecture with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 7)
        )

    def constrain_parameters(self, x):
        # res_size: integer in range
        res_size = torch.clamp(x[:, 0], self.res_size_range[0], self.res_size_range[1])
        res_size = torch.round(res_size)

        # k: integer in range
        k = torch.clamp(x[:, 1], self.k_range[0], self.k_range[1])
        k = torch.round(k)

        # p: probability between 0 and 1
        p = torch.sigmoid(x[:, 2])

        # m: integer in range
        m = torch.clamp(x[:, 3], self.m_range[0], self.m_range[1])
        m = torch.round(m)

        # alpha: probability between 0 and 1
        alpha = torch.sigmoid(x[:, 4])

        # weight scales: positive values with reasonable range
        small_weight_scale = torch.exp(torch.clamp(x[:, 5], -5, 2))
        scale_weight_scale = torch.exp(torch.clamp(x[:, 6], -5, 2))

        return torch.stack([res_size, k, p, m, alpha,
                            small_weight_scale, scale_weight_scale], dim=1)

    def forward(self, x):
        raw_output = self.network(x)
        return self.constrain_parameters(raw_output)


def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)

        # Remove any rows with infinite or NaN values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # Separate features and target
        X = data[['nrmse']].values
        y = data[['res_size', 'k', 'p', 'm', 'alpha',
                  'small_weight_scale', 'scale_weight_scale']].values

        # Custom scaling
        X_scaler = RobustStandardScaler()
        y_scaler = RobustStandardScaler()

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test_scaled)

        return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                X_scaler, y_scaler)

    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise


def train_model(model, X_train, y_train, epochs=1000, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 100
    min_lr = 1e-6

    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"Invalid loss value at epoch {epoch}")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Learning rate scheduling
            scheduler.step(loss)

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch}")
                break

            if optimizer.param_groups[0]['lr'] < min_lr:
                print(f"Learning rate too small at epoch {epoch}")
                break

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


def find_optimal_parameters(model, y_scaler, target_nrmse=0.0):
    try:
        model.eval()
        with torch.no_grad():
            target = torch.FloatTensor([[target_nrmse]])
            scaled_params = model(target)
            optimal_params = y_scaler.inverse_transform(scaled_params.numpy())

            # Ensure parameters are within valid ranges
            params_dict = {
                'resSize': max(10, min(1000, int(np.round(optimal_params[0, 0])))),
                'k': max(2, min(50, int(np.round(optimal_params[0, 1])))),
                'p': max(0, min(1, float(optimal_params[0, 2]))),
                'm': max(2, min(50, int(np.round(optimal_params[0, 3])))),
                'alpha': max(0, min(1, float(optimal_params[0, 4]))),
                'small_weight_scale': max(0, float(optimal_params[0, 5])),
                'scale_weight_scale': max(0, float(optimal_params[0, 6]))
            }

            return params_dict

    except Exception as e:
        print(f"Error in parameter optimization: {str(e)}")
        return None


def main():
    try:
        # Load and preprocess data
        X_train, y_train, X_test, y_test, X_scaler, y_scaler = load_and_preprocess_data(
            'combined_n==5,20,50_all_random_seed_NARMA.csv'
        )

        # Create and train model
        model = ESNParameterPredictor()
        train_model(model, X_train, y_train)

        # Fiشnd optimal parameters
        optimal_params = find_optimal_parameters(model, y_scaler, target_nrmse=0.0)

        if optimal_params:
            print("\nOptimal ESN parameters for NRMSE = 0:")
            for param, value in optimal_params.items():
                print(f"{param} = {value}")
        else:
            print("Could not find valid parameters.")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()
