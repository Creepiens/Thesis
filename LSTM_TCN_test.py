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

###################### forme des ANN ######################

# Définition du modèle LSTM
class LSTMModel_1(nn.Module):
    def __init__(self):
        super(LSTMModel_1, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 2)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Définition du modèle TCN
class TCNModel_1(nn.Module):
    def __init__(self):
        super(TCNModel_1, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
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

###################### entrainement & evaluation ######################

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

def train_model_with_reinjection(model, optimizer, X_tensor, y_tensor, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for i in range(sample_length):
            optimizer.zero_grad()
            output = model(X_tensor[i].unsqueeze(0))
            loss = criterion(output, y_tensor[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

            # Réinjection des valeurs prédites
            if i < len(X_tensor) - 1:
                X_tensor[i+1][0][-2:] = output.detach()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Fonction d'évaluation avec réinjection
def evaluate_model(model, X_tensor):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_tensor)):
            output = model(X_tensor[i].unsqueeze(0))
            predictions.append(output)

            # Réinjection des valeurs prédites
            if i < len(X_tensor) - 1:
                X_tensor[i+1][0][-2:] = output

    return torch.stack(predictions).squeeze()


# Calcul des métriques de performance
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

###################### génération des données ######################

C1, C2, h = 1.0, 1.0, 1  # Exemple de valeurs pour C1, C2, h
methodes = ["sin", "random", "lhs"]
acuraty = 10 # nombre de point par seconde
# Conditions initiales
T1_0 = 0.0
T2_0 = 0.0

# Génération des données
def generate_data(method='sin', size=60):
    t = np.linspace(0, duration, size)
    if method == 'sin':
        P1 = np.sin(t)
        P2 = np.cos(3*t)
    elif method == 'random':
        P1 = np.random.rand(size)*2-1
        P2 = np.random.rand(size)*2-1
    elif method == 'lhs':
        sampler = qmc.LatinHypercube(d=2)
        sample = sampler.random(n=size)
        P1, P2 = sample[:, 0]*2-1, sample[:, 1]*2-1

    # Simulation du système
    def system(t, y):
        T1, T2 = y
        index = min(int(t / duration * size), size - 1)  # Assure que l'index ne dépasse pas size - 1
        dT1_dt = (P1[index] - h * (T1 - T2)) / C1
        dT2_dt = (P2[index] + h * (T1 - T2)) / C2
        return [dT1_dt, dT2_dt]

    y0 = [T1_0, T2_0]
    sol = solve_ivp(system, [0, duration], y0, t_eval=t, vectorized=True)
    T1, T2 = sol.y

    return T1, T2, P1, P2, t


# génération des donnée :
duration = 60
T1_sim_val, T2_sim_val, P1_values_val, P2_values_val, t_eval_val = generate_data(method='sin', size= int(duration*acuraty))
duration = 360
T1_sim_train, T2_sim_train, P1_values_train, P2_values_train, t_eval_train = generate_data(method='random', size=int(duration*acuraty))

###################### afficher données ######################
if True :
    for methode in methodes :
        T1, T2, P1, P2, t = generate_data(method=methode, size=1000)
        plt.figure(figsize=(12, 6))
        plt.plot(t, P1, label='Excitation P1', linestyle='--', color='cyan')
        plt.plot(t, P2, label='Excitation P2', linestyle='--', color='yellow')
        plt.plot(t, T1, label='T1 simulé', color='blue')
        plt.plot(t, T2, label='T2 simulé', color='orange')
        plt.xlabel('Temps (s)')
        plt.ylabel('Valeurs')
        plt.title(f'Évolution des températures T1 et T2 avec la méthode : {methode}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Visualisation des données simulées pour la validation :
    plt.figure(figsize=(12, 6))
    plt.plot(t_eval_val, P1_values_val, label='Excitation P1', linestyle='--', color='blue')
    plt.plot(t_eval_val, P2_values_val, label='Excitation P2', linestyle='--', color='orange')
    plt.plot(t_eval_val, T1_sim_val, label='T1 simulé', color='blue')
    plt.plot(t_eval_val, T2_sim_val, label='T2 simulé', color='orange')
    plt.xlabel('Temps (s)')
    plt.ylabel('Valeurs')
    plt.title('Évolution des températures T1 et T2 avec les excitations validation')
    plt.legend()
    plt.grid(True)
    plt.show()

###################### mise en forme des données ######################

# Longueur de l'échantillon
sample_length = 20

# Préparation des données pour l'entraînement de ANN
X_train = np.vstack((P1_values_train, P2_values_train, T1_sim_train, T2_sim_train)).T
y_train = np.vstack((T1_sim_train, T2_sim_train)).T

X_val = np.vstack((P1_values_val, P2_values_val, T1_sim_val, T2_sim_val)).T
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

show = ["lstm", "tcn", "lstm2"]

criterion = nn.MSELoss()
if "lstm" in show :
    model_lstm = LSTMModel_1()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)
    train_model(model_lstm, optimizer_lstm, X_tensor_train, y_tensor_train, num_epochs=100)
    predicted_lstm_train = evaluate_model(model_lstm, X_tensor_train)
    predicted_lstm_val = evaluate_model(model_lstm, X_tensor_val)
    
if "tcn" in show :
    model_tcn = TCNModel_1()
    optimizer_tcn = optim.Adam(model_tcn.parameters(), lr=0.001)
    train_model(model_tcn, optimizer_tcn, X_tensor_train, y_tensor_train, num_epochs=100)
    predicted_tcn_train = evaluate_model(model_tcn, X_tensor_train)
    predicted_tcn_val = evaluate_model(model_tcn, X_tensor_val)
    
if "lstm2" in show :
    model_lstm_2 = LSTMModel_1()
    optimizer_lstm_2 = optim.Adam(model_lstm_2.parameters(), lr=0.001)
    train_model_with_reinjection(model_lstm_2, optimizer_lstm_2, X_tensor_train, y_tensor_train, num_epochs=100)
    predicted_lstm_train_2 = evaluate_model(model_lstm_2, X_tensor_train)
    predicted_lstm_val_2 = evaluate_model(model_lstm_2, X_tensor_val)

###################### Visualisation résultat : ######################
visualisation = ["ecart", "absolue"][1]

# Métriques pour les données d'entraînement
mse_lstm_train, rmse_lstm_train, mae_lstm_train, r2_lstm_train = calculate_metrics(y_tensor_train.numpy(), predicted_lstm_train.numpy())
mse_tcn_train, rmse_tcn_train, mae_tcn_train, r2_tcn_train = calculate_metrics(y_tensor_train.numpy(), predicted_tcn_train.numpy())

# Métriques pour les données de validation
mse_lstm_val, rmse_lstm_val, mae_lstm_val, r2_lstm_val = calculate_metrics(y_tensor_val.numpy(), predicted_lstm_val.numpy())
mse_tcn_val, rmse_tcn_val, mae_tcn_val, r2_tcn_val = calculate_metrics(y_tensor_val.numpy(), predicted_tcn_val.numpy())

# Affichage des métriques
if False :
    print("Métriques pour les données d'entraînement (LSTM):")
    print(f"MSE: {mse_lstm_train:.4f}, RMSE: {rmse_lstm_train:.4f}, MAE: {mae_lstm_train:.4f}, R²: {r2_lstm_train:.4f}")
    print("Métriques pour les données d'entraînement (TCN):")
    print(f"MSE: {mse_tcn_train:.4f}, RMSE: {rmse_tcn_train:.4f}, MAE: {mae_tcn_train:.4f}, R²: {r2_tcn_train:.4f}")
    print("Métriques pour les données de validation (LSTM):")
    print(f"MSE: {mse_lstm_val:.4f}, RMSE: {rmse_lstm_val:.4f}, MAE: {mae_lstm_val:.4f}, R²: {r2_lstm_val:.4f}")
    print("Métriques pour les données de validation (TCN):")
    print(f"MSE: {mse_tcn_val:.4f}, RMSE: {rmse_tcn_val:.4f}, MAE: {mae_tcn_val:.4f}, R²: {r2_tcn_val:.4f}")


if "lstm" in show:
    if visualisation == "ecart" :
        error_lstm_train_T1 = T1_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)] - predicted_lstm_train[:, 0].numpy()
        error_lstm_train_T2 = T2_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)] - predicted_lstm_train[:, 1].numpy()
    elif visualisation == "absolue" :
        error_lstm_train_T1 = predicted_lstm_train[:, 0].numpy()
        error_lstm_train_T2 = predicted_lstm_train[:, 1].numpy()
        
if "tcn" in show:
    if visualisation == "ecart" :
        error_lstm_train_2_T1 = T1_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)] - predicted_lstm_train_2[:, 0].numpy()
        error_lstm_train_2_T2 = T2_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)] - predicted_lstm_train_2[:, 1].numpy()
    elif visualisation == "absolue" :
        error_lstm_train_2_T1 = predicted_lstm_train_2[:, 0].numpy()
        error_lstm_train_2_T2 = predicted_lstm_train_2[:, 1].numpy()

if "lstm2" in show:
    if visualisation == "ecart" :
        error_tcn_train_T1 = T1_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)] - predicted_tcn_train[:, 0].numpy()
        error_tcn_train_T2 = T2_sim_train[sample_length-1:sample_length-1+len(y_tensor_train)] - predicted_tcn_train[:, 1].numpy()
    elif visualisation == "absolue" :
        error_tcn_train_T1 = predicted_tcn_train[:, 0].numpy()
        error_tcn_train_T2 = predicted_tcn_train[:, 1].numpy()
            
# Visualisation des écarts sur les données d'entraînement
plt.figure(figsize=(12, 6))
if "lstm" in show:
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], error_lstm_train_T1, label='T1 (LSTM train)', linestyle='-.', color='blue')
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], error_lstm_train_T2, label='T2 (LSTM train)', linestyle='--', color='cyan')
if "tcn" in show:
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], error_lstm_train_2_T1, label='T1 (LSTM train_2)', linestyle='-.', color='black')
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], error_lstm_train_2_T2, label='T2 (LSTM train_2)', linestyle='--', color='grey')
if "lstm2" in show:
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], error_tcn_train_T1, label='T1 (TCN train)', linestyle='-.', color='orange')
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], error_tcn_train_T2, label='T2 (TCN train)', linestyle='--', color='red')
if visualisation == "ecart" :
    plt.title("Écarts sur les données d'entrainement")
    plt.ylabel('Écart')
elif visualisation == "absolue" :    
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], y_tensor_train[:,0] , label='valeur réel T1', linestyle='-', color='blue')
    plt.plot(t_eval_train[sample_length-1:sample_length-1+len(y_tensor_train)], y_tensor_train[:,1] , label='valeur réel T2', linestyle='-', color='orange')    
    plt.title("Valeur absolue sur les données d'entrainement")
    plt.ylabel('valeur absolue')
plt.xlabel('Temps (s)')
plt.legend()
plt.show()

# Calcul des écarts pour les données de validation
if "lstm" in show:
    if visualisation == "ecart" :
        error_lstm_val_T1 = T1_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)] - predicted_lstm_val[:, 0].numpy()
        error_lstm_val_T2 = T2_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)] - predicted_lstm_val[:, 1].numpy()
    elif visualisation == "absolue" :
        error_lstm_val_T1 = predicted_lstm_val[:, 0].numpy()
        error_lstm_val_T2 = predicted_lstm_val[:, 1].numpy()

if "tcn" in show:
    if visualisation == "ecart" :
        error_lstm_val_2_T1 = T1_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)] - predicted_lstm_val_2[:, 0].numpy()
        error_lstm_val_2_T2 = T2_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)] - predicted_lstm_val_2[:, 1].numpy()
    elif visualisation == "absolue" :
        error_lstm_val_2_T1 = predicted_lstm_val_2[:, 0].numpy()
        error_lstm_val_2_T2 = predicted_lstm_val_2[:, 1].numpy()

if "lstm2" in show:
    if visualisation == "ecart" :
        error_tcn_val_T1 = T1_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)] - predicted_tcn_val[:, 0].numpy()
        error_tcn_val_T2 = T2_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)] - predicted_tcn_val[:, 1].numpy()
    elif visualisation == "absolue" :
        error_tcn_val_T1 = predicted_tcn_val[:, 0].numpy()
        error_tcn_val_T2 = predicted_tcn_val[:, 1].numpy()
        
        
# Visualisation des écarts sur les données de validation
plt.figure(figsize=(12, 6))
if "lstm" in show:
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], error_lstm_val_T1, label='T1 (LSTM val)', linestyle='-.', color='blue')
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], error_lstm_val_T2, label='T2 (LSTM val)', linestyle='--', color='cyan')
if "tcn" in show:
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], error_lstm_val_2_T1, label='T1 (LSTM val_2)', linestyle='-.', color='black')
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], error_lstm_val_2_T2, label='T2 (LSTM val_2)', linestyle='--', color='grey')
if "lstm2" in show:
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], error_tcn_val_T1, label='T1 (TCN val)', linestyle='-.', color='orange')
   
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], error_tcn_val_T2, label='T2 (TCN val)', linestyle='--', color='red')
if visualisation == "ecart" :
    plt.title('Écarts sur les données de validation')
    plt.ylabel('Écart')
elif visualisation == "absolue" :
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], y_tensor_val[:,0] , label='valeur réel T1', linestyle='-', color='blue')
    plt.plot(t_eval_val[sample_length-1:sample_length-1+len(y_tensor_val)], y_tensor_val[:,1] , label='valeur réel T2', linestyle='-', color='orange')
    plt.ylabel('valeur aboslue')
    plt.title('Valeur absolue sur les données de validation')

plt.xlabel('Temps (s)')
plt.legend()
plt.show()
