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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import random

###################### forme des ANN ######################

# Définition du modèle RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 30)  # Première couche entièrement connectée
        self.dropout = nn.Dropout(0.5)  # Couche de dropout
        self.fc2 = nn.Linear(30, output_size)  # Deuxième couche entièrement connectée
        self.batchnorm = nn.BatchNorm1d(30)  # Couche de normalisation par lots

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h_0)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.batchnorm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=X_tensor_train.shape[2], hidden_size=50, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(50, 30)  # Première couche entièrement connectée
        self.dropout = nn.Dropout(0.5)  # Couche de dropout
        self.fc2 = nn.Linear(30, y_tensor_train.shape[1])  # Deuxième couche entièrement connectée
        self.batchnorm = nn.BatchNorm1d(30)  # Couche de normalisation par lots

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))

        # Prenez la sortie de la dernière étape de la séquence
        out = out[:, -1, :]

        # Appliquez les couches supplémentaires
        out = self.fc1(out)
        out = self.batchnorm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

# Définition du modèle TCN
class TCNModel_1(nn.Module):
    def __init__(self):
        super(TCNModel_1, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.tcn(x)
        return out[:, :, -1]

###################### entrainement & evaluation ######################

# Entraînement des modèles
def train_model(model, optimizer, X_train, y_train, num_epochs=200):
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
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
                X_tensor[i+1, int(X_tensor.shape[1]/2):, -1] = output
    return torch.stack(predictions).squeeze()


# Calcul des métriques de performance
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

###################### génération des données V2 ######################

methodes = ["sin", "random", "lhs", "pseudo-random"]

def generate_datas(num_variables, method='sin', duration=60, frequency=60):
    t = np.linspace(1, duration, duration)

    # Génération des P_i
    if method == 'sin':
        P = [(np.sin(t / (i+1)*1000) + 1) / 2 for i in range(num_variables)]
    elif method == 'random':
        P = [np.random.rand(duration) for _ in range(num_variables)]
    elif method == 'lhs':
        sampler = qmc.LatinHypercube(d=num_variables)
        sample = sampler.random(n=duration)
        P = [sample[:, i] for i in range(num_variables)]
    elif method == 'pseudo-random':
        P = [np.random.choice([0, 1], size=duration) for _ in range(num_variables)]
    elif method == 'flat':
        P = [np.repeat(np.array([1, int(random.choice([0, 1])), int(random.choice([0, 1])), 0]), int(duration / 4)) for _ in range(num_variables)]

    if frequency > 1:
        P = [np.repeat(p, frequency) for p in P]
    elif frequency < 1:
        num_samples = int(duration * frequency)
        indices = np.linspace(0, duration - 1, num=num_samples, dtype=int)
        P = [p[indices] for p in P]

    P = [p * 100 for p in P]

    t = np.linspace(1, duration, int(duration * frequency))

    # Génération des coefficients h_ij
    h = np.random.rand(num_variables, num_variables)

    # Simulation du système
    def system(t, y, k=1):
        dT_dt = np.zeros_like(y)
        C = random.random()
        for i in range(num_variables):
            dT_dt[i] = (P[i][int(round(t * frequency) - 1)] - k * y[i]) / C
            for j in range(num_variables):
                if i != j:
                    dT_dt[i] += h[i, j] * (y[j] - y[i]) / C
        return dT_dt

    k = 1
    y0 = np.ones(num_variables)
    sol = solve_ivp(system, [0, duration], y0, t_eval=t, vectorized=True)
    T = sol.y
    return T, P, t

###################### génération des données ######################

if True :
    # génération des donnée :
    frequency = 5
    num_variables = 4
    
    duration = 60
    T_sim_val, P_sim_val, t_sim_val = generate_datas(num_variables, method='flat', duration=duration, frequency= int(frequency))
    duration = 360
    T_sim_train, P_sim_train, t_sim_train = generate_datas(num_variables, method='pseudo-random', duration=duration, frequency= int(frequency))

###################### afficher données ######################
if False :
    for methode in methodes :
        color_map = plt.get_cmap('tab10')
        frequency= 0.5
        T_s, P_s, t = generate_datas(method=methode, duration=1000, frequency= frequency)
        plt.figure(figsize=(12, 6))
        for i, T in enumerate(T_s) :
            plt.plot(t_sim_val, T, label=f'Température T{i}', linestyle='-', color=color_map(i))
        for i, P in enumerate(P_s) :
            plt.plot(t_sim_val, P, label=f'Excitation P{i}', linestyle='-.', color=color_map(i))
        plt.xlabel('Temps (s)')
        plt.ylabel('Valeurs')
        plt.title(f'Évolution des températures T1 et T2 avec la méthode : {methode}, à frequence {frequency}Hz')
        plt.legend()
        plt.grid(True)
        plt.show()

if True :
    color_map = plt.get_cmap('tab10')
    plt.figure(figsize=(12, 6))
    for i, T in enumerate(T_sim_train) :
        plt.plot(t_sim_train, T, label=f'Température T{i}', linestyle='-', color=color_map(i))
    #for i, P in enumerate(P_sim_train) :
    #    plt.plot(t_sim_train, P, label=f'Excitation P{i}', linestyle='-.', color=color_map(i))
    plt.xlabel('Temps (s)')
    plt.ylabel('Valeurs')
    plt.title(f'Évolution des températures T1 et T2 avec les exitation entrainement')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualisation des données simulées pour la validation :
    plt.figure(figsize=(12, 6))
    for i, T in enumerate(T_sim_val) :
        plt.plot(t_sim_val, T, label=f'Température T{i}', linestyle='-', color=color_map(i))
    for i, P in enumerate(P_sim_val) :
        plt.plot(t_sim_val, P, label=f'Excitation P{i}', linestyle='-.', color=color_map(i))
    plt.xlabel('Temps (s)')
    plt.ylabel('Valeurs')
    plt.title('Évolution des températures T1 et T2 avec les excitations validation')
    plt.legend()
    plt.grid(True)
    plt.show()

###################### mise en forme des données ######################

# Longueur de l'échantillon
sample_length = 10

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train = np.concatenate((np.vstack(P_sim_train), np.vstack(T_sim_train)), axis=0)
X_train *= 0.01
y_train = np.column_stack(T_sim_train).T
y_train *= 0.01

X_val = np.concatenate((np.vstack(P_sim_val), np.vstack(T_sim_val)), axis=0)
X_val *= 0.01
y_val = np.column_stack(T_sim_val).T
y_val *= 0.01

# Création des séquences d'échantillons
X_seq_train = np.array([X_train[:, i:i+sample_length] for i in range(X_train.shape[1]-sample_length+1)])
y_seq_train = np.array([y_train[:, i+sample_length-1] for i in range(y_train.shape[1]-sample_length+1)])

X_seq_val = np.array([X_val[:, i:i+sample_length] for i in range(X_val.shape[1]-sample_length+1)])
y_seq_val = np.array([y_val[:, i+sample_length-1] for i in range(y_val.shape[1]-sample_length+1)])

# Conversion en tenseurs PyTorch
X_tensor_train = torch.tensor(X_seq_train, dtype=torch.float32)
y_tensor_train = torch.tensor(y_seq_train, dtype=torch.float32)

X_tensor_val = torch.tensor(X_seq_val, dtype=torch.float32)
y_tensor_val = torch.tensor(y_seq_val, dtype=torch.float32)

show = ["lstm", "tcn", "rnn"]


criterion = nn.MSELoss()
if "rnn" in show :
    model_rnn = LSTMModel()
    optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=0.001)
    train_model(model_rnn, optimizer_rnn, X_tensor_train, y_tensor_train, num_epochs=1000)
    predicted_rnn_train = evaluate_model(model_rnn, X_tensor_train)
    predicted_rnn_val = evaluate_model(model_rnn, X_tensor_val)
    
if "tcn" in show :
    model_tcn = TCNModel_1()
    optimizer_tcn = optim.Adam(model_tcn.parameters(), lr=0.001)
    train_model(model_tcn, optimizer_tcn, X_tensor_train, y_tensor_train, num_epochs=500)
    predicted_tcn_train = evaluate_model(model_tcn, X_tensor_train)
    predicted_tcn_val = evaluate_model(model_tcn, X_tensor_val)
    
if "lstm" in show :
    model_lstm = LSTMModel()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)
    train_model(model_lstm, optimizer_lstm, X_tensor_train, y_tensor_train, num_epochs=500)
    predicted_lstm_train = evaluate_model(model_lstm, X_tensor_train)
    predicted_lstm_val= evaluate_model(model_lstm, X_tensor_val)

###################### Visualisation résultat : ######################
visualisations = ["ecart", "absolue"]


# Affichage des métriques
if False :
    # Métriques pour les données d'entraînement
    mse_lstm_train, rmse_lstm_train, mae_lstm_train, r2_lstm_train = calculate_metrics(y_tensor_train.numpy(), predicted_lstm_train.numpy())
    mse_tcn_train, rmse_tcn_train, mae_tcn_train, r2_tcn_train = calculate_metrics(y_tensor_train.numpy(), predicted_tcn_train.numpy())

    # Métriques pour les données de validation
    mse_lstm_val, rmse_lstm_val, mae_lstm_val, r2_lstm_val = calculate_metrics(y_tensor_val.numpy(), predicted_lstm_val.numpy())
    mse_tcn_val, rmse_tcn_val, mae_tcn_val, r2_tcn_val = calculate_metrics(y_tensor_val.numpy(), predicted_tcn_val.numpy())

    print("Métriques pour les données d'entraînement (LSTM):")
    print(f"MSE: {mse_lstm_train:.4f}, RMSE: {rmse_lstm_train:.4f}, MAE: {mae_lstm_train:.4f}, R²: {r2_lstm_train:.4f}")
    print("Métriques pour les données d'entraînement (TCN):")
    print(f"MSE: {mse_tcn_train:.4f}, RMSE: {rmse_tcn_train:.4f}, MAE: {mae_tcn_train:.4f}, R²: {r2_tcn_train:.4f}")
    print("Métriques pour les données de validation (LSTM):")
    print(f"MSE: {mse_lstm_val:.4f}, RMSE: {rmse_lstm_val:.4f}, MAE: {mae_lstm_val:.4f}, R²: {r2_lstm_val:.4f}")
    print("Métriques pour les données de validation (TCN):")
    print(f"MSE: {mse_tcn_val:.4f}, RMSE: {rmse_tcn_val:.4f}, MAE: {mae_tcn_val:.4f}, R²: {r2_tcn_val:.4f}")

def inverse_transform(predictions):
    return predictions*100

for visualisation in visualisations :
    color_map = plt.get_cmap('tab10')
    plt.figure(figsize=(12, 6))
    if "lstm" in show:
        for i, T in enumerate(predicted_lstm_val.T) :
            if visualisation == "ecart" :
                T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
            T = inverse_transform(T)
            plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (LSTM train)', linestyle='--', color=color_map(i))
        mse_lstm_val, rmse_lstm_val, mae_lstm_val, r2_lstm_val = calculate_metrics(y_tensor_val.numpy(), predicted_lstm_val.numpy())
        print(f"Métriques pour les données de validation (LSTM): \n MSE: {mse_lstm_val:.4f}, RMSE: {rmse_lstm_val:.4f}, MAE: {mae_lstm_val:.4f}, R²: {r2_lstm_val:.4f}")

    if "tcn" in show:
        for i, T in enumerate(predicted_tcn_val.T) :
            if visualisation == "ecart" :
                T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
            T = inverse_transform(T)
            plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (TCN train)', linestyle='-.', color=color_map(i))
        mse_tcn_val, rmse_tcn_val, mae_tcn_val, r2_tcn_val = calculate_metrics(y_tensor_val.numpy(), predicted_tcn_val.numpy())
        print(f"Métriques pour les données de validation (TCN): \n MSE: {mse_tcn_val:.4f}, RMSE: {rmse_tcn_val:.4f}, MAE: {mae_tcn_val:.4f}, R²: {r2_tcn_val:.4f}")

    if "rnn" in show:
        for i, T in enumerate(predicted_rnn_val.T) :
            if visualisation == "ecart" :
                T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
            T = inverse_transform(T)
            plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (rnn train)', linestyle=':', color=color_map(i))
        mse_rnn_val, rmse_rnn_val, mae_rnn_val, r2_rnn_val = calculate_metrics(y_tensor_val.numpy(), predicted_rnn_val.numpy())
        print(f"Métriques pour les données de validation (RNN): \n MSE: {mse_rnn_val:.4f}, RMSE: {rmse_rnn_val:.4f}, MAE: {mae_rnn_val:.4f}, R²: {r2_rnn_val:.4f}")
    
    if visualisation == "ecart" :
        plt.title("écart des températuress sur les données de validation")
        plt.ylabel('écart des températures')
    elif visualisation == "absolue" :    
        for i, T in enumerate(T_sim_val) :
            plt.plot(t_sim_val, T, label=f'Température T{i}', linestyle='-', color=color_map(i))        
        plt.title("Température sur les données de validation")
        plt.ylabel('Température')
        
    plt.xlabel('Temps (s)')
    plt.legend()
    plt.show()
