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
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tcn import TCN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import time
import optuna
from optuna.trial import TrialState

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
def train_model(model, optimizer, train_loader, valid_dataset, DEVICE, trial=None, num_epochs=200):
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        
        metrics = evaluate_model(model, valid_dataset)
        accuracy = metrics[3]
        if trial is not None:
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return accuracy
        
# Fonction d'évaluation avec réinjection
def evaluate_model(model, valid_dataset, export_result=False):
    X_tensor, y_tensor = valid_dataset
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(X_tensor)):
            output = model(X_tensor[i].unsqueeze(0))
            predictions.append(output)
            # Réinjection des valeurs prédites
            if i < len(X_tensor) - 1:
                X_tensor[i+1, -1, :-int(X_tensor.shape[2]/2)] = output
    metrics = calculate_metrics(y_tensor, torch.stack(predictions).squeeze())    
    if export_result == True :
        return torch.stack(predictions).squeeze(), *metrics
    return metrics


# Calcul des métriques de performance
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

###################### génération des données ######################

def generate_datas(num_variables, method='sin', duration=90, frequency=60, duration_job=1, statique=True):
    t = np.linspace(1, int(duration/duration_job), int(duration/duration_job))
    
    if method == 'sin':
        P = [(np.sin(t / (i+1)*1000) + 1) / 2 for i in range(num_variables)]
    elif method == 'random':
        P = [np.random.rand(int(duration/duration_job)) for _ in range(num_variables)]
    elif method == 'lhs':
        sampler = qmc.LatinHypercube(d=num_variables)
        sample = sampler.random(n=int(duration/duration_job))
        P = [sample[:, i] for i in range(num_variables)]
    elif method == 'pseudo-random':
        P = [np.random.choice([0, 1], size=int(duration/duration_job)) for _ in range(num_variables)]
    elif method == 'flat':
        P = [np.repeat(np.array([int(random.choice([0, 1])), 1, round(random.random(), 1), round(random.random(), 1) ,int(random.choice([0, 1])), 0]), int(duration/duration_job/ 6)) for _ in range(num_variables)]

    if frequency > 1:
        P = [np.repeat(p, frequency) for p in P]
    elif frequency < 1:
        num_samples = int(duration * frequency)
        indices = np.linspace(0, duration - 1, num=num_samples, dtype=int)
        P = [p[indices] for p in P]

    if duration_job > 1 :
        P = [np.repeat(p, duration_job) for p in P]

    P = [p * 100 for p in P]

    t = np.linspace(1, duration, int(duration * frequency))

    # Génération des coefficients h_ij
    h = np.random.rand(num_variables, num_variables)#/100

    # Simulation du système
    def system(t, y, k=1/100):
        dT_dt = np.zeros_like(y)
        C = random.random()*500 #Capacité à stoker l'énergie
        for i in range(num_variables):
            dT_dt[i] = (P[i][int(round(t * frequency) - 1)] - k * y[i]**2) / C
            for j in range(num_variables):
                if i != j:
                    dT_dt[i] += h[i, j] * (y[j] - y[i]) / C
        return dT_dt

    y0 = np.ones(num_variables)*50 #valeur de base
    sol = solve_ivp(system, [0, duration], y0, t_eval=t, vectorized=True)
    T = sol.y
    return T, P, t

# mise en forme des données :
def prepare_data_set(P_sim, T_sim, sample_length, BATCHSIZE):
    X = np.concatenate((np.vstack(P_sim), np.vstack(T_sim)), axis=0)
    X *= 0.01 #Standardisation pour être proche de [0,1] pour facilité l'entrainent
    y = np.column_stack(T_sim).T
    y *= 0.01 #Standardisation pour être proche de [0,1] pour facilité l'entrainent

    # Création des séquences d'échantillons
    X_seq = np.array([X[:, i:i+sample_length] for i in range(X.shape[1]-sample_length+1)])
    y_seq = np.array([y[:, i+sample_length-1] for i in range(y.shape[1]-sample_length+1)])

    # Conversion en tenseurs PyTorch
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    X_tensor = X_tensor.permute(0, 2, 1)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)

    dataset = [X_tensor, y_tensor]
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCHSIZE, shuffle=False)
    
    return X, y, X_seq, y_seq, X_tensor, y_tensor, dataset, loader


# Optuna simulation :
def objective(trial) : 
    ###################### choix des hyperparamètre ######################
    #Construction du data set :
    num_variables = 4
    frequency = 1
    duration_val = 90
    BATCHSIZE = 128
    methode_train = trial.suggest_categorical("methode_train", ["sin", "random", "lhs", "pseudo-random"])
    duration_train = trial.suggest_int("duration_train", 60, 3600)
    
    #Variable pour la construction des modèles :
    sample_length = trial.suggest_int("sample_length", 1, 30)
    sample_length //= frequency
    epochs = trial.suggest_int("epochs", 100, 1000)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    #Type de modèle :
    construct_model = [trial.suggest_categorical("construct_model", ["LSTM", "CNN", "RNN", "TCN"])]
    print(construct_model)
    
    #Paramètre machine :
    DEVICE = torch.device("cpu")

    ###################### génération des données ######################
    # génération des donnée et prépration:
    T_sim_val, P_sim_val, t_sim_val = generate_datas(num_variables, method='flat', duration=duration_val, duration_job=300, frequency= int(frequency))
    X_val, y_val, X_seq_val, y_seq_val, X_tensor_val, y_tensor_val, valid_dataset, valid_loader = prepare_data_set(P_sim_val, T_sim_val, sample_length ,BATCHSIZE)

    T_sim_train, P_sim_train, t_sim_train = generate_datas(num_variables, method=methode_train, duration=duration_train, duration_job=300, frequency= int(frequency))
    X_train, y_train, X_seq_train, y_seq_train, X_tensor_train, y_tensor_train, train_dataset, train_loader = prepare_data_set(P_sim_train, T_sim_train, sample_length ,BATCHSIZE)
 
    ###################### construction modèles ######################
    criterion = nn.MSELoss()
    if "TCN" in construct_model:
        model_TCN = TCNModel(X_seq_train.shape[2], y_seq_train.shape[1]).to(DEVICE)
        optimizer_TCN = getattr(optim, optimizer_name)(model_TCN.parameters(), lr=lr)
        accuracy = train_model(model_TCN, optimizer_TCN, train_loader, valid_dataset, DEVICE, trial=trial, num_epochs=epochs)
        mse_TCN_val, rmse_TCN_val, mae_TCN_val, r2_TCN_val = evaluate_model(model_TCN, valid_dataset)
        print(f"Métriques pour les données de validation (TCN): \n MSE: {mse_TCN_val:.4f}, RMSE: {rmse_TCN_val:.4f}, MAE: {mae_TCN_val:.4f}, R²: {r2_TCN_val:.4f}")

    if "CNN" in construct_model:
        model_CNN = CNNModel(X_seq_train.shape[2], y_seq_train.shape[1]).to(DEVICE)
        optimizer_CNN = getattr(optim, optimizer_name)(model_CNN.parameters(), lr=lr)
        accuracy = train_model(model_CNN, optimizer_CNN, train_loader, valid_dataset, DEVICE, trial=trial, num_epochs=epochs)
        mse_CNN_val, rmse_CNN_val, mae_CNN_val, r2_CNN_val = evaluate_model(model_CNN, valid_dataset)
        print(f"Métriques pour les données de validation (CNN): \n MSE: {mse_CNN_val:.4f}, RMSE: {rmse_CNN_val:.4f}, MAE: {mae_CNN_val:.4f}, R²: {r2_CNN_val:.4f}")

    if "RNN" in construct_model:
        model_RNN = LSTMModel(X_seq_train.shape[1], y_seq_train.shape[1]).to(DEVICE)
        optimizer_RNN = getattr(optim, optimizer_name)(model_RNN.parameters(), lr=lr)
        accuracy = train_model(model_RNN, optimizer_RNN, train_loader, valid_dataset, DEVICE, trial=trial, num_epochs=epochs)
        mse_RNN_val, rmse_RNN_val, mae_RNN_val, r2_RNN_val = evaluate_model(model_RNN, valid_dataset)
        print(f"Métriques pour les données de validation (RNN): \n MSE: {mse_RNN_val:.4f}, RMSE: {rmse_RNN_val:.4f}, MAE: {mae_RNN_val:.4f}, R²: {r2_RNN_val:.4f}")
        
    if "LSTM" in construct_model:
        model_LSTM = LSTMModel(X_seq_train.shape[1], y_seq_train.shape[1]).to(DEVICE)
        optimizer_LSTM = getattr(optim, optimizer_name)(model_LSTM.parameters(), lr=lr)
        accuracy = train_model(model_LSTM, optimizer_LSTM, train_loader, valid_dataset, DEVICE, trial=trial, num_epochs=epochs)
        mse_LSTM_val, rmse_LSTM_val, mae_LSTM_val, r2_LSTM_val = evaluate_model(model_LSTM, valid_dataset)
        print(f"Métriques pour les données de validation (LSTM): \n MSE: {mse_LSTM_val:.4f}, RMSE: {rmse_LSTM_val:.4f}, MAE: {mae_LSTM_val:.4f}, R²: {r2_LSTM_val:.4f}")
        
    #return [accuracy, duration]
    return accuracy

# Test unitaire d'entrainement et validation
def run_a_test(methode_train, epochs=200, lr=0.001, optimizer_name = "Adam",  construct_model= ["LSTM", "CNN", "RNN", "TCN"], 
               sample_length = 5 , duration_train = 21600, duration_val = 3600, frequency = 1, num_variables = 4, 
               visualisations = ["ecart", "absolue"]):
    
    ###################### paramétrage : ######################
    #Construction du data set :
    BATCHSIZE = 128
    sample_length //= frequency
    DEVICE = torch.device("cpu")
    
    ###################### sauvegarde des résultat ######################
    results = {}
    for model in construct_model:
        results[model] = {"duration":None, "MSE": None, "RMSE":None, "MAE":None, "R²":None} 
    
    ###################### génération des données ######################
    
    # génération des donnée et prépration:
    T_sim_val, P_sim_val, t_sim_val = generate_datas(num_variables, method='flat', duration=duration_val, duration_job=300, frequency= int(frequency))
    X_val, y_val, X_seq_val, y_seq_val, X_tensor_val, y_tensor_val, valid_dataset, valid_loader = prepare_data_set(P_sim_val, T_sim_val, sample_length ,BATCHSIZE)

    T_sim_train, P_sim_train, t_sim_train = generate_datas(num_variables, method=methode_train, duration=duration_train, duration_job=300, frequency= int(frequency))
    X_train, y_train, X_seq_train, y_seq_train, X_tensor_train, y_tensor_train, train_dataset, train_loader = prepare_data_set(P_sim_train, T_sim_train, sample_length ,BATCHSIZE)
    
    ###################### afficher données ######################
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
    
 
    ###################### construction modèles ######################
    criterion = nn.MSELoss()
    if "TCN" in construct_model:
        start = time.time()
        model_TCN = TCNModel(X_seq_train.shape[2], y_seq_train.shape[1]).to(DEVICE)
        optimizer_TCN = getattr(optim, optimizer_name)(model_TCN.parameters(), lr=lr)
        accuracy = train_model(model_TCN, optimizer_TCN, train_loader, valid_dataset, DEVICE, num_epochs=epochs)
        duration = time.time()-start
        results["TCN"]["duration"] = duration
        print(f"TCN time train : {duration:.1f} secondes")
        predicted_TCN_val, mse_TCN_val, rmse_TCN_val, mae_TCN_val, r2_TCN_val = evaluate_model(model_TCN, valid_dataset, export_result=True)
        results["TCN"] = {"MSE": rmse_TCN_val, "RMSE":mae_TCN_val, "MAE":mae_TCN_val, "R²":r2_TCN_val}
        print(f"Métriques pour les données de validation (TCN): \n MSE: {mse_TCN_val:.4f}, RMSE: {rmse_TCN_val:.4f}, MAE: {mae_TCN_val:.4f}, R²: {r2_TCN_val:.4f}")

    if "CNN" in construct_model:
        start = time.time()
        model_CNN = CNNModel(X_seq_train.shape[2], y_seq_train.shape[1]).to(DEVICE)
        optimizer_CNN = getattr(optim, optimizer_name)(model_CNN.parameters(), lr=lr)
        accuracy = train_model(model_CNN, optimizer_CNN, train_loader, valid_dataset, DEVICE, num_epochs=epochs)
        duration = time.time()-start
        results["CNN"]["duration"] = duration
        print(f"CNN time train : {duration:.1f} secondes")
        predicted_CNN_val, mse_CNN_val, rmse_CNN_val, mae_CNN_val, r2_CNN_val = evaluate_model(model_CNN, valid_dataset, export_result=True)
        results["CNN"] = {"MSE": rmse_CNN_val, "RMSE":mae_CNN_val, "MAE":mae_CNN_val, "R²":r2_CNN_val}
        print(f"Métriques pour les données de validation (CNN): \n MSE: {mse_CNN_val:.4f}, RMSE: {rmse_CNN_val:.4f}, MAE: {mae_CNN_val:.4f}, R²: {r2_CNN_val:.4f}")

    if "RNN" in construct_model:
        start = time.time()
        model_RNN = LSTMModel(X_seq_train.shape[1], y_seq_train.shape[1]).to(DEVICE)
        optimizer_RNN = getattr(optim, optimizer_name)(model_RNN.parameters(), lr=lr)
        accuracy = train_model(model_RNN, optimizer_RNN, train_loader, valid_dataset, DEVICE, num_epochs=epochs)
        duration = time.time()-start
        results["RNN"]["duration"] = duration
        print(f"RNN time train : {duration:.1f} secondes")
        predicted_RNN_val, mse_RNN_val, rmse_RNN_val, mae_RNN_val, r2_RNN_val = evaluate_model(model_RNN, valid_dataset, export_result=True)
        results["RNN"] = {"MSE": rmse_RNN_val, "RMSE":mae_RNN_val, "MAE":mae_RNN_val, "R²":r2_RNN_val}
        print(f"Métriques pour les données de validation (RNN): \n MSE: {mse_RNN_val:.4f}, RMSE: {rmse_RNN_val:.4f}, MAE: {mae_RNN_val:.4f}, R²: {r2_RNN_val:.4f}")
        
    if "LSTM" in construct_model:
        start = time.time()
        model_LSTM = LSTMModel(X_seq_train.shape[1], y_seq_train.shape[1]).to(DEVICE)
        optimizer_LSTM = getattr(optim, optimizer_name)(model_LSTM.parameters(), lr=lr)
        accuracy = train_model(model_LSTM, optimizer_LSTM, train_loader, valid_dataset, DEVICE, num_epochs=epochs)
        duration = time.time()-start
        results["LSTM"]["duration"] = duration
        print(f"LSTM time train : {duration:.1f} secondes")
        #predicted_LSTM_train = evaluate_model(model_LSTM, X_tensor_train)
        predicted_LSTM_val, mse_LSTM_val, rmse_LSTM_val, mae_LSTM_val, r2_LSTM_val = evaluate_model(model_LSTM, valid_dataset, export_result=True)
        results["LSTM"] = {"MSE": rmse_LSTM_val, "RMSE":mae_LSTM_val, "MAE":mae_LSTM_val, "R²":r2_LSTM_val}
        print(f"Métriques pour les données de validation (LSTM): \n MSE: {mse_LSTM_val:.4f}, RMSE: {rmse_LSTM_val:.4f}, MAE: {mae_LSTM_val:.4f}, R²: {r2_LSTM_val:.4f}")

    ###################### Visualisation résultat : ######################   
    #Standardisation pour retourner entre 0 et 100.
    def inverse_transform(predictions):
        return predictions*100
    
    for visualisation in visualisations :
        color_map = plt.get_cmap('tab10')
        plt.figure(figsize=(12, 6))
        if "LSTM" in construct_model:
            for i, T in enumerate(predicted_LSTM_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (LSTM train)', linestyle='--', color=color_map(i))

        if "CNN" in construct_model:
            for i, T in enumerate(predicted_CNN_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (CNN train)', linestyle='-.', color=color_map(i))

        if "RNN" in construct_model:
            for i, T in enumerate(predicted_RNN_val.T) :
                if visualisation == "ecart" :
                    T = T.numpy() - y_val[i, sample_length-1:sample_length-1+len(y_tensor_train)]
                T = inverse_transform(T)
                plt.plot(t_sim_val[sample_length-1:sample_length-1+len(y_tensor_val)], T, label=f'T{i} (RNN train)', linestyle=':', color=color_map(i))
        
        if "TCN" in construct_model:
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


if __name__ == "__main__":
    #Choix de test automatique ou manuel !
    manual = True
    
    if manual == True :
        methode_train =["flat", "sin", "random", "pseudo-random", "lhs"]
        results = run_a_test(methode_train[2], construct_model= ["CNN"])
        print(results)
        
    elif manual == False :
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=3600)
    
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
    
        print("Best trial:")
        trial = study.best_trial
    
        print("  Value: ", trial.value)
    
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
