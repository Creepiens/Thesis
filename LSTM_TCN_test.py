# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:00:41 2025

@author: grljbeur
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# Paramètres du système
C1 = 1
C2 = 1
h = 0.5
methodes = ["sin", "random", "lhs"]

# Fonction pour les équations différentielles
def system(t, T, P1, P2):
    T1, T2 = T
    dT1_dt = (P1(t) - h * (T1 - T2)) / C1
    dT2_dt = (P2(t) + h * (T1 - T2)) / C2
    return [dT1_dt, dT2_dt]

# Générer des échantillons LHS
def generate_lhs_samples(n_samples, n_features):
    sampler = qmc.LatinHypercube(d=n_features)
    sample = sampler.random(n=n_samples)
    l_bounds = [-1, -1]
    u_bounds = [1, 1]
    scale = qmc.scale(sample, l_bounds, u_bounds)
    return scale

# Forçages externes
def P1(t, lhs_samples):
    global excitation_methode, t_span
    if excitation_methode == "sin":
        return np.sin(t)
    elif excitation_methode == "random":
        return (random.random() - 0.5) * 2
    elif excitation_methode == "lhs":
        index = int(np.round((t - t_span[0]) / (t_span[1] - t_span[0]) * (len(lhs_samples) - 1)))  # Calculer l'index
        return lhs_samples[index, 0]


def P2(t, lhs_samples):
    global excitation_methode, t_span
    if excitation_methode == "sin":
        return np.cos(3 * t)
    elif excitation_methode == "random":
        return (random.random() - 0.5) * 2
    elif excitation_methode == "lhs":
        index = int(np.round((t - t_span[0]) / (t_span[1] - t_span[0]) * (len(lhs_samples) - 1)))  # Calculer l'index
        return lhs_samples[index, 1]

# Conditions initiales
T1_0 = 0.0
T2_0 = 0.0

acuracy = 6

# Temps de simulation
t_span_train = (0, 60)
t_eval_train = np.linspace(*t_span_train, (t_span_train[1]-t_span_train[0])*acuracy)

## Choix de la métode de génération des données entrainement ##
excitation_methode = methodes[2]
t_span = t_span_train
lhs_samples_train = generate_lhs_samples((t_span_train[1]-t_span_train[0])*acuracy, 2)  # Échantillons pour l'entraînement
sol_train = solve_ivp(system, t_span_train, [T1_0, T2_0], t_eval=t_eval_train, args=(lambda t: P1(t, lhs_samples_train), lambda t: P2(t, lhs_samples_train)))

# Données simulées pour l'entraînement
T1_sim_train = sol_train.y[0]
T2_sim_train = sol_train.y[1]

# Calcul des excitations P1 et P2 pour chaque instant de t_eval
P1_values_train = np.array([P1(t, lhs_samples_train) for t in t_eval_train])
P2_values_train = np.array([P2(t, lhs_samples_train) for t in t_eval_train])

# Temps pour la validation
t_span_val = (60, 70)
t_eval_val = np.linspace(*t_span_val, (t_span_val[1]-t_span_val[0])*acuracy)  

## Choix de la métode de génération des données validation ##
excitation_methode = methodes[2]
t_span = t_span_val
lhs_samples_val = generate_lhs_samples((t_span_val[1]-t_span_val[0])*acuracy, 2)
sol_val = solve_ivp(system, t_span_val, [T1_sim_train[-1], T2_sim_train[-1]], t_eval=t_eval_val, args=(lambda t: P1(t, lhs_samples_val), lambda t: P2(t, lhs_samples_val)))

# Données simulées pour la validation
T1_sim_val = sol_val.y[0]
T2_sim_val = sol_val.y[1]

# Calcul des excitations P1 et P2 pour chaque instant de t_eval
P1_values_val = np.array([P1(t, lhs_samples_val) for t in t_eval_val])
P2_values_val = np.array([P2(t, lhs_samples_val) for t in t_eval_val])

# Visualisation des données simulées avec les excitations
plt.figure(figsize=(12, 6))
plt.plot(t_eval_train, T1_sim_train, label='T1 simulé', color='blue')
plt.plot(t_eval_train, T2_sim_train, label='T2 simulé', color='orange')
plt.plot(t_eval_train, P1_values_train, label='Excitation P1', linestyle='--', color='green')
plt.plot(t_eval_train, P2_values_train, label='Excitation P2', linestyle='--', color='red')
plt.xlabel('Temps (s)')
plt.ylabel('Valeurs')
plt.title('Évolution des températures T1 et T2 avec les excitations entrainement')
plt.legend()
plt.grid(True)
plt.show()


# Visualisation des données simulées avec les excitations
plt.figure(figsize=(12, 6))
plt.plot(t_eval_val, T1_sim_val, label='T1 simulé', color='blue')
plt.plot(t_eval_val, T2_sim_val, label='T2 simulé', color='orange')
plt.plot(t_eval_val, P1_values_val, label='Excitation P1', linestyle='--', color='green')
plt.plot(t_eval_val, P2_values_val, label='Excitation P2', linestyle='--', color='red')
plt.xlabel('Temps (s)')
plt.ylabel('Valeurs')
plt.title('Évolution des températures T1 et T2 avec les excitations validation')
plt.legend()
plt.grid(True)
plt.show()

# Longueur de l'échantillon
sample_length = 10

# Préparation des données pour l'entraînement du RN
X_train = np.vstack((P1_values_train, P2_values_train)).T
y_train = np.vstack((T1_sim_train, T2_sim_train)).T

X_val = np.vstack((P1_values_val, P2_values_val)).T
y_val = np.vstack((T1_sim_val, T2_sim_val)).T

# Création des séquences d'échantillons
X_seq_train = [X_train[i:i+sample_length] for i in range(len(X_train)-sample_length+1)]
y_seq_train = [y_train[i+sample_length-1] for i in range(len(y_train)-sample_length+1)]

X_seq_val = [X_val[i:i+sample_length] for i in range(len(X_val)-sample_length+1)]
y_seq_val = [y_val[i+sample_length-1] for i in range(len(y_val)-sample_length+1)]

# Conversion en tenseurs PyTorch
X_tensor_train = torch.tensor(X_seq_train, dtype=torch.float32)
y_tensor_train = torch.tensor(y_seq_train, dtype=torch.float32)

X_tensor_val = torch.tensor(X_seq_val, dtype=torch.float32)
y_tensor_val = torch.tensor(y_seq_val, dtype=torch.float32)

# Définition du modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 2)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Définition du modèle TCN
class TCNModel(nn.Module):
    def __init__(self):
        super(TCNModel, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Changer la forme pour Conv1d
        out = self.tcn(x)
        out = out.permute(0, 2, 1)  # Changer la forme pour la sortie
        return out[:, -1, :]

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model_lstm = LSTMModel()
model_tcn = TCNModel()
criterion = nn.MSELoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)
optimizer_tcn = optim.Adam(model_tcn.parameters(), lr=0.001)

# Entraînement des modèles
def train_model(model, optimizer, X_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Entraînement du modèle LSTM
train_model(model_lstm, optimizer_lstm, X_tensor_train, y_tensor_train, num_epochs=100)

# Entraînement du modèle TCN
train_model(model_tcn, optimizer_tcn, X_tensor_train, y_tensor_train, num_epochs=100)

# Évaluation des modèles
def evaluate_model(model, X):
    model.eval()
    with torch.no_grad():
        predicted = model(X)
    return predicted

# Prédictions des modèles sur les données d'entraînement et de validation
predicted_lstm_train = evaluate_model(model_lstm, X_tensor_train)
predicted_tcn_train = evaluate_model(model_tcn, X_tensor_train)
predicted_lstm_val = evaluate_model(model_lstm, X_tensor_val)
predicted_tcn_val = evaluate_model(model_tcn, X_tensor_val)

# Calcul des métriques de performance
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# Métriques pour les données d'entraînement
mse_lstm_train, rmse_lstm_train, mae_lstm_train, r2_lstm_train = calculate_metrics(y_tensor_train.numpy(), predicted_lstm_train.numpy())
mse_tcn_train, rmse_tcn_train, mae_tcn_train, r2_tcn_train = calculate_metrics(y_tensor_train.numpy(), predicted_tcn_train.numpy())

# Métriques pour les données de validation
mse_lstm_val, rmse_lstm_val, mae_lstm_val, r2_lstm_val = calculate_metrics(y_tensor_val.numpy(), predicted_lstm_val.numpy())
mse_tcn_val, rmse_tcn_val, mae_tcn_val, r2_tcn_val = calculate_metrics(y_tensor_val.numpy(), predicted_tcn_val.numpy())

# Affichage des métriques
print("Métriques pour les données d'entraînement (LSTM):")
print(f"MSE: {mse_lstm_train:.4f}, RMSE: {rmse_lstm_train:.4f}, MAE: {mae_lstm_train:.4f}, R²: {r2_lstm_train:.4f}")
print("Métriques pour les données d'entraînement (TCN):")
print(f"MSE: {mse_tcn_train:.4f}, RMSE: {rmse_tcn_train:.4f}, MAE: {mae_tcn_train:.4f}, R²: {r2_tcn_train:.4f}")
print("Métriques pour les données de validation (LSTM):")
print(f"MSE: {mse_lstm_val:.4f}, RMSE: {rmse_lstm_val:.4f}, MAE: {mae_lstm_val:.4f}, R²: {r2_lstm_val:.4f}")
print("Métriques pour les données de validation (TCN):")
print(f"MSE: {mse_tcn_val:.4f}, RMSE: {rmse_tcn_val:.4f}, MAE: {mae_tcn_val:.4f}, R²: {r2_tcn_val:.4f}")

# Visualisation des prédictions sur les données d'entraînement
plt.figure(figsize=(12, 6))
plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], T1_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)], label='T1 simulé (train)', color='blue')
plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], T2_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)], label='T2 simulé (train)', color='orange')
plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], predicted_lstm_train[:, 0].numpy(), label='T1 prédit (LSTM train)', linestyle='--', color='blue')
plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], predicted_lstm_train[:, 1].numpy(), label='T2 prédit (LSTM train)', linestyle='--', color='orange')
plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], predicted_tcn_train[:, 0].numpy(), label='T1 prédit (TCN train)', linestyle='--', color='green')
plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], predicted_tcn_train[:, 1].numpy(), label='T2 prédit (TCN train)', linestyle='--', color='red')
plt.legend()
plt.title('Prédictions sur les données d\'entraînement')
plt.show()

# Visualisation des prédictions sur les données de validation
plt.figure(figsize=(12, 6))
plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], T1_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], label='T1 simulé (val)', color='blue')
plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], T2_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], label='T2 simulé (val)', color='orange')
plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], predicted_lstm_val[:, 0].numpy(), label='T1 prédit (LSTM val)', linestyle='--', color='blue')
plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], predicted_lstm_val[:, 1].numpy(), label='T2 prédit (LSTM val)', linestyle='--', color='orange')
plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], predicted_tcn_val[:, 0].numpy(), label='T1 prédit (TCN val)', linestyle='--', color='green')
plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], predicted_tcn_val[:, 1].numpy(), label='T2 prédit (TCN val)', linestyle='--', color='red')
plt.legend()
plt.title('Prédictions sur les données de validation')
plt.show()
