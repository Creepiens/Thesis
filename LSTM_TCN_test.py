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
from pytorch_tcn import TCN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import time

###################### forme des ANN ######################

# Définition du modèle RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.RNN = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 30)  # Première couche entièrement connectée
        self.dropout = nn.Dropout(0.5)  # Couche de dropout
        self.fc2 = nn.Linear(30, output_size)  # Deuxième couche entièrement connectée
        self.batchnorm = nn.BatchNorm1d(30)  # Couche de normalisation par lots

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.RNN(x, h_0)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.batchnorm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    
# Définition du modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMModel, self).__init__()
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=50, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(50, 30)  # Première couche entièrement connectée
        self.dropout = nn.Dropout(0.5)  # Couche de dropout
        self.fc2 = nn.Linear(30, output_size)  # Deuxième couche entièrement connectée
        self.batchnorm = nn.BatchNorm1d(30)  # Couche de normalisation par lots

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        out, _ = self.LSTM(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.batchnorm(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

# Définition du modèle CNN
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=output_size, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.CNN(x)
        return out[:, :, -1]
    
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels= [32, 64, 64], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        out = self.tcn(x)
        out = self.linear(out[:, :, -1])
        return out

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
                #X_tensor[i+1, int(X_tensor.shape[1]/2):, -1] = output
                X_tensor[i+1, -1, :-int(X_tensor.shape[1]/2)+1] = output
    return torch.stack(predictions).squeeze()


# Calcul des métriques de performance
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

###################### génération des données V2 ######################

def generate_datas(num_variables, method='sin', duration=60, frequency=60, statique=True):
    t = np.linspace(1, duration, duration)

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
        P = [np.repeat(np.array([1, int(random.choice([0, 1])), 0.5 ,int(random.choice([0, 1])), 0]), int(duration / 5)) for _ in range(num_variables)]

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


def run_test(methode_train, sample_length = 10, epochs=200, show = ["LSTM", "CNN", "RNN", "TCN"], duration_train = 360, duration_val = 60, frequency = 5, num_variables = 4,    visualisations = ["ecart", "absolue"]):
    ###################### sauvegarde des résultat ######################
    results = {}
    for model in show :
        results[model] = {"duration":None, "MSE": None, "RMSE":None, "MAE":None, "R²":None} 
    
    ###################### génération des données ######################
    
    # génération des donnée :
    T_sim_val, P_sim_val, t_sim_val = generate_datas(num_variables, method='flat', duration=duration_val, frequency= int(frequency))
    T_sim_train, P_sim_train, t_sim_train = generate_datas(num_variables, method=methode_train, duration=duration_train, frequency= int(frequency))
    
    ###################### afficher données ######################
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

    X_train = np.concatenate((np.vstack(P_sim_train), np.vstack(T_sim_train)), axis=0)
    X_train *= 0.01 #Standardisation pour être proche de [0,1] pour facilité l'entrainent
    y_train = np.column_stack(T_sim_train).T
    y_train *= 0.01 #Standardisation pour être proche de [0,1] pour facilité l'entrainent
    
    X_val = np.concatenate((np.vstack(P_sim_val), np.vstack(T_sim_val)), axis=0)
    X_val *= 0.01 #Standardisation pour être proche de [0,1] pour facilité l'entrainent
    y_val = np.column_stack(T_sim_val).T
    y_val *= 0.01 #Standardisation pour être proche de [0,1] pour facilité l'entrainent
    
    # Création des séquences d'échantillons
    X_seq_train = np.array([X_train[:, i:i+sample_length] for i in range(X_train.shape[1]-sample_length+1)])
    y_seq_train = np.array([y_train[:, i+sample_length-1] for i in range(y_train.shape[1]-sample_length+1)])
    
    X_seq_val = np.array([X_val[:, i:i+sample_length] for i in range(X_val.shape[1]-sample_length+1)])
    y_seq_val = np.array([y_val[:, i+sample_length-1] for i in range(y_val.shape[1]-sample_length+1)])
    
    # Conversion en tenseurs PyTorch
    X_tensor_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_tensor_train = X_tensor_train.permute(0, 2, 1)
    y_tensor_train = torch.tensor(y_seq_train, dtype=torch.float32)

    X_tensor_val = torch.tensor(X_seq_val, dtype=torch.float32)
    X_tensor_val = X_tensor_val.permute(0, 2, 1)
    y_tensor_val = torch.tensor(y_seq_val, dtype=torch.float32)
    
    ###################### construction modèles ######################
    
    criterion = nn.MSELoss()
    if "TCN" in show :
        start = time.time()
        model_TCN = TCNModel(X_seq_train.shape[2], y_seq_train.shape[1])
        optimizer_TCN = optim.Adam(model_TCN.parameters(), lr=0.001)
        train_model(model_TCN, optimizer_TCN, X_tensor_train, y_tensor_train, num_epochs=epochs)
        duration = time.time()-start
        results["TCN"] = {"duration":duration}
        print(f"TCN time train : {duration:.1f} secondes")
        #predicted_TCN_train = evaluate_model(model_TCN, X_tensor_train)
        predicted_TCN_val= evaluate_model(model_TCN, X_tensor_val)

    if "CNN" in show :
        start = time.time()
        model_CNN = CNNModel(X_seq_train.shape[2], y_seq_train.shape[1])
        optimizer_CNN = optim.Adam(model_CNN.parameters(), lr=0.001)
        train_model(model_CNN, optimizer_CNN, X_tensor_train, y_tensor_train, num_epochs=epochs)
        duration = time.time()-start
        results["CNN"] = {"duration":duration}
        print(f"CNN time train : {duration:.1f} secondes")
        #predicted_CNN_train = evaluate_model(model_CNN, X_tensor_train)
        predicted_CNN_val = evaluate_model(model_CNN, X_tensor_val)
            
    if "RNN" in show :
        start = time.time()
        model_RNN = LSTMModel(X_seq_train.shape[1], y_seq_train.shape[1])
        optimizer_RNN = optim.Adam(model_RNN.parameters(), lr=0.001)
        train_model(model_RNN, optimizer_RNN, X_tensor_train, y_tensor_train, num_epochs=epochs)
        duration = time.time()-start
        results["RNN"] = {"duration":duration}
        print(f"RNN time train : {duration:.1f} secondes")
        #predicted_RNN_train = evaluate_model(model_RNN, X_tensor_train)
        predicted_RNN_val = evaluate_model(model_RNN, X_tensor_val)
        
    if "LSTM" in show :
        start = time.time()
        model_LSTM = LSTMModel(X_seq_train.shape[1], y_seq_train.shape[1])
        optimizer_LSTM = optim.Adam(model_LSTM.parameters(), lr=0.001)
        train_model(model_LSTM, optimizer_LSTM, X_tensor_train, y_tensor_train, num_epochs=epochs)
        duration = time.time()-start
        results["LSTM"] = {"duration":duration}
        print(f"LSTM time train : {duration:.1f} secondes")
        #predicted_LSTM_train = evaluate_model(model_LSTM, X_tensor_train)
        predicted_LSTM_val= evaluate_model(model_LSTM, X_tensor_val)
        
    ###################### Calcules des métriques : ######################
    mse_LSTM_val, rmse_LSTM_val, mae_LSTM_val, r2_LSTM_val = calculate_metrics(y_tensor_val.numpy(), predicted_LSTM_val.numpy())
    results["LSTM"] = {"MSE": rmse_LSTM_val, "RMSE":mae_LSTM_val, "MAE":mae_LSTM_val, "R²":r2_LSTM_val}
    print(f"Métriques pour les données de validation (LSTM): \n MSE: {mse_LSTM_val:.4f}, RMSE: {rmse_LSTM_val:.4f}, MAE: {mae_LSTM_val:.4f}, R²: {r2_LSTM_val:.4f}")
    mse_CNN_val, rmse_CNN_val, mae_CNN_val, r2_CNN_val = calculate_metrics(y_tensor_val.numpy(), predicted_CNN_val.numpy())
    results["CNN"] = {"MSE": rmse_CNN_val, "RMSE":mae_CNN_val, "MAE":mae_CNN_val, "R²":r2_CNN_val}
    print(f"Métriques pour les données de validation (CNN): \n MSE: {mse_CNN_val:.4f}, RMSE: {rmse_CNN_val:.4f}, MAE: {mae_CNN_val:.4f}, R²: {r2_CNN_val:.4f}")
    mse_RNN_val, rmse_RNN_val, mae_RNN_val, r2_RNN_val = calculate_metrics(y_tensor_val.numpy(), predicted_RNN_val.numpy())
    results["RNN"] = {"MSE": rmse_RNN_val, "RMSE":mae_RNN_val, "MAE":mae_RNN_val, "R²":r2_RNN_val}
    print(f"Métriques pour les données de validation (RNN): \n MSE: {mse_RNN_val:.4f}, RMSE: {rmse_RNN_val:.4f}, MAE: {mae_RNN_val:.4f}, R²: {r2_RNN_val:.4f}")
    mse_TCN_val, rmse_TCN_val, mae_TCN_val, r2_TCN_val = calculate_metrics(y_tensor_val.numpy(), predicted_TCN_val.numpy())
    results["TCN"] = {"MSE": rmse_TCN_val, "RMSE":mae_TCN_val, "MAE":mae_TCN_val, "R²":r2_TCN_val}
    print(f"Métriques pour les données de validation (TCN): \n MSE: {mse_TCN_val:.4f}, RMSE: {rmse_TCN_val:.4f}, MAE: {mae_TCN_val:.4f}, R²: {r2_TCN_val:.4f}")
    
    ###################### Visualisation résultat : ######################   
    #Standardisation pour retourner entre 0 et 100.
    def inverse_transform(predictions):
        return predictions*100
    
    for visualisation in visualisations :
        color_map = plt.get_cmap('tab10')
        plt.figure(figsize=(12, 6))
        if "LSTM" in show:
            for i, T in enumerate(predicted_LSTM_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (LSTM train)', linestyle='--', color=color_map(i))

        if "CNN" in show:
            for i, T in enumerate(predicted_CNN_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (CNN train)', linestyle='-.', color=color_map(i))

        if "RNN" in show:
            for i, T in enumerate(predicted_RNN_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (RNN train)', linestyle=':', color=color_map(i))
        
        if "TCN" in show:
            for i, T in enumerate(predicted_TCN_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (TCN train)', linestyle=(0, (3, 1, 1, 1, 1, 1)), color=color_map(i))
        
        if visualisation == "ecart" :
            plt.plot(t_sim_val, np.linspace(0, 0, len(t_sim_val)), label="True value", linestyle='-', color='black')
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
        
    return results

if __name__ == '__main__':
    methodes = ["sin", "random", "lhs", "pseudo-random"]
    results = run_test("pseudo-random")
    print(results)