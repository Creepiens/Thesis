# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:00:41 2025

@author: grljbeur
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random

# Paramètres du système
C1 = 1.0
C2 = 5.0
h = 0.5
rdm = False

# Fonction pour les équations différentielles
def system(t, T, P1, P2):
    T1, T2 = T
    dT1_dt = (P1(t) - h * (T1 - T2)) / C1
    dT2_dt = (P2(t) + h * (T1 - T2)) / C2
    return [dT1_dt, dT2_dt]

# Forçages externes
def P1(t):
    return np.sin(t)

def P2(t):
    global rdm
    if rdm == False :
        return np.cos(t)
    else :
        if np.isscalar(t):
            return random.random() - 0.5
        else:
            return np.array([random.random() - 0.5 for _ in t])

# Conditions initiales
T1_0 = 0.0
T2_0 = 0.0

# Temps de simulation
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Résolution du système
sol = solve_ivp(system, t_span, [T1_0, T2_0], t_eval=t_eval, args=(P1, P2))

# Données simulées
T1_sim = sol.y[0]
T2_sim = sol.y[1]

# Visualisation des données simulées
plt.plot(t_eval, T1_sim, label='T1 simulé')
plt.plot(t_eval, T2_sim, label='T2 simulé')
plt.legend()
plt.show()

# Longueur de l'échantillon
sample_length = 50

# Préparation des données pour l'entraînement du RN
X = np.vstack((P1(t_eval), P2(t_eval))).T
y = np.vstack((T1_sim, T2_sim)).T

# Création des séquences d'échantillons
X_seq = [X[i:i+sample_length] for i in range(len(X)-sample_length+1)]
y_seq = [y[i+sample_length-1] for i in range(len(y)-sample_length+1)]

# Conversion en tenseurs PyTorch
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

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
train_model(model_lstm, optimizer_lstm, X_train, y_train, num_epochs=100)

# Entraînement du modèle TCN
train_model(model_tcn, optimizer_tcn, X_train, y_train, num_epochs=100)

# Évaluation des modèles
def evaluate_model(model, X):
    model.eval()
    with torch.no_grad():
        predicted = model(X)
    return predicted

# Prédictions des modèles sur les données d'entraînement et de test
predicted_lstm_train = evaluate_model(model_lstm, X_train)
predicted_tcn_train = evaluate_model(model_tcn, X_train)
predicted_lstm_test = evaluate_model(model_lstm, X_test)
predicted_tcn_test = evaluate_model(model_tcn, X_test)

# Visualisation des prédictions sur les données d'entraînement
plt.figure(figsize=(12, 6))
plt.plot(t_eval[sample_length-1:sample_length-1+len(y_train)], y_train[:, 0].numpy(), label='T1 simulé (train)', color='blue')
plt.plot(t_eval[sample_length-1:sample_length-1+len(y_train)], y_train[:, 1].numpy(), label='T2 simulé (train)', color='orange')
plt.plot(t_eval[sample_length-1:sample_length-1+len(y_train)], predicted_lstm_train[:, 0].numpy(), label='T1 prédit (LSTM train)', linestyle='--', color='blue')
plt.plot(t_eval[sample_length-1:sample_length-1+len(y_train)], predicted_lstm_train[:, 1].numpy(), label='T2 prédit (LSTM train)', linestyle='--', color='orange')
plt.plot(t_eval[sample_length-1:sample_length-1+len(y_train)], predicted_tcn_train[:, 0].numpy(), label='T1 prédit (TCN train)', linestyle='--', color='green')
plt.plot(t_eval[sample_length-1:sample_length-1+len(y_train)], predicted_tcn_train[:, 1].numpy(), label='T2 prédit (TCN train)', linestyle='--', color='red')
plt.legend()
plt.title('Prédictions sur les données d\'entraînement')
plt.show()

# Visualisation des prédictions sur les données de test
plt.figure(figsize=(12, 6))
plt.plot(t_eval[sample_length-1+len(y_train):sample_length-1+len(y_train)+len(y_test)], y_test[:, 0].numpy(), label='T1 simulé (test)', color='blue')
plt.plot(t_eval[sample_length-1+len(y_train):sample_length-1+len(y_train)+len(y_test)], y_test[:, 1].numpy(), label='T2 simulé (test)', color='orange')
plt.plot(t_eval[sample_length-1+len(y_train):sample_length-1+len(y_train)+len(y_test)], predicted_lstm_test[:, 0].numpy(), label='T1 prédit (LSTM test)', linestyle='--', color='blue')
plt.plot(t_eval[sample_length-1+len(y_train):sample_length-1+len(y_train)+len(y_test)], predicted_lstm_test[:, 1].numpy(), label='T2 prédit (LSTM test)', linestyle='--', color='orange')
plt.plot(t_eval[sample_length-1+len(y_train):sample_length-1+len(y_train)+len(y_test)], predicted_tcn_test[:, 0].numpy(), label='T1 prédit (TCN test)', linestyle='--', color='green')
plt.plot(t_eval[sample_length-1+len(y_train):sample_length-1+len(y_train)+len(y_test)], predicted_tcn_test[:, 1].numpy(), label='T2 prédit (TCN test)', linestyle='--', color='red')
plt.legend()
plt.title('Prédictions sur les données de test')
plt.show()

