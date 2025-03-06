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
        self.res_size_range = (10, 400)  # Reasonable range for reservoir size
        self.k_range = (2, 12)  # Range for k
        self.m_range = (2, 12)  # Range for m

        # Network architecture with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(7, 256),
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

            nn.Linear(256, 1)
        )

    def constrain_parameters(self, x):
        # No parameter constraints needed for NRMSE output
        return x

    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)

        # Remove any rows with infinite or NaN values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # Separate features and target (swapped from original)
        X = data[['res_size', 'k', 'p', 'm', 'alpha',
                  'small_weight_scale', 'scale_weight_scale']].values
        y = data[['nrmse']].values

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


def find_optimal_parameters(model, X_scaler, y_scaler, target_params):
    try:
        model.eval()
        with torch.no_grad():
            # Scale the input parameters
            scaled_params = X_scaler.transform(np.array([target_params]))
            scaled_params_tensor = torch.FloatTensor(scaled_params)

            # Predict NRMSE
            predicted_nrmse = model(scaled_params_tensor)

            # Convert back to original scale
            nrmse = y_scaler.inverse_transform(predicted_nrmse.numpy())

            return float(nrmse[0, 0])

    except Exception as e:
        print(f"Error in NRMSE prediction: {str(e)}")
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
        model.eval()  # Set to evaluation mode
        # Define your upper and lower bounds
        lower_bounds = torch.tensor([[50.0, 3.0, 0.10, 3.0, 0.10, 0.10, 0.10]])
        upper_bounds = torch.tensor([[500.0, 10.0, 0.50, 10.0, 0.90, 0.90, 0.90]])
        # Initial tensor
        input_tensor = torch.tensor([[212.0, 8.0, 0.2, 8.0, 0.49, 0.51, 0.50]], requires_grad=True)
        # Define the optimizer
        optimizer = optim.SGD([input_tensor], lr=0.01, momentum=0.9)

        # List to store loss values for plotting
        loss_values = []
        iterations = []

        # Optimization loop
        num_iterations = 100
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = output * output
            loss.backward()
            optimizer.step()

            # Store loss value for plotting
            loss_values.append(loss.item())
            iterations.append(iteration)

            # Clamp values AFTER optimizer step
            if iteration > 5:
                with torch.no_grad():
                    input_tensor.data = torch.clamp(input_tensor.data,
                                                    min=lower_bounds,
                                                    max=upper_bounds)
            # Print progress
            if iteration % 10 == 0:
                print(f"{iteration},{output.item()}")
                print(f"{input_tensor.detach().numpy()}")

        # Final optimized input (guaranteed to be within bounds)
        optimized_inputs = input_tensor.detach().numpy()
        print('-------------------------------')
        print("resSize = ", int(optimized_inputs[0, 0]))
        print("k = ", int(optimized_inputs[0, 1]))
        print("p = ", optimized_inputs[0, 2])
        print("m = ", int(optimized_inputs[0, 3]))
        print("alpha = ", optimized_inputs[0, 4])
        print("small_weight_scale = ", optimized_inputs[0, 5])
        print("scale_weight_scale = ", optimized_inputs[0, 6])

        # Plot the loss values
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, loss_values, 'b-', label='Loss (output²)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        plt.title('Optimization Progress: Loss vs Iteration')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')  # Using log scale for better visualization
        plt.show()

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    main()