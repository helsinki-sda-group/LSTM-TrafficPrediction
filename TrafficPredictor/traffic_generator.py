import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from datetime import datetime
import joblib

import json
import os

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, activation_function):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim[0], 1, batch_first = True
        ).to(device)

        self.lstm2 = nn.LSTM(
            hidden_dim[0], hidden_dim[1], 1, batch_first = True
        ).to(device)

        self.dropout = nn.Dropout(p = dropout_prob)

        if activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim[1], output_dim).to(device)


    def forward(self, x):
        h0_1 = torch.zeros(1, x.size(0), self.hidden_dim[0]).requires_grad_()
        h0_1 = h0_1.to(device)
        c0_1 = torch.zeros(1, x.size(0), self.hidden_dim[0]).requires_grad_()
        c0_1 = c0_1.to(device)

        out_1, (hn_1, cn_1) = self.lstm1(x, (h0_1.detach(), c0_1.detach()))
        out_1 = self.activation(out_1)
        out_1 = self.dropout(out_1)

        h0_2 = torch.zeros(1, out_1.size(0), self.hidden_dim[1]).requires_grad_()
        h0_2 = h0_2.to(device)
        c0_2 = torch.zeros(1, out_1.size(0), self.hidden_dim[1]).requires_grad_()
        c0_2 = c0_2.to(device)

        out_2, (hn_2, cn_2) = self.lstm2(out_1, (h0_2.detach(), c0_2.detach()))
        out = self.fc(out_2)

        return out


def get_model(model, model_params):
    models = {
        "lstm": LSTMModel
    }
    return models.get(model.lower())(**model_params)

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def evaluate(self, test_loader, batch_size = 1, n_features = 1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().cpu().numpy())
                values.append(y_test.to(device).detach().cpu().numpy())

        return predictions, values

    def predict(self, data):
        with torch.no_grad():
            data = data.view([data.shape[0], -1, data.shape[1]]).to(device)
            self.model.eval()
            yhat = self.model(data)
            predictions = yhat.to(device).detach().numpy()
        return predictions


def load_model(model_params, model_path, optimizer_arg, learning_rate, weight_decay):
    model = get_model('lstm', model_params)
    model.load_state_dict(torch.load(model_path))

    loss_fn = nn.MSELoss(reduction = "mean")
    if optimizer_arg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    elif optimizer_arg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

    print('Set {} as optimizer'.format(optimizer_arg))

    return Optimization(model = model, loss_fn = loss_fn, optimizer = optimizer)    



def preprocess_generated_values(generated_values):
    generated_values[generated_values < 0] = 0
    return np.round(generated_values)

def get_day_of_week(year, month, day):
    return datetime(year, month, day).weekday() + 1

def process_time(time):
    time = str(int(time))
    if float(time) < 10:
        time = '0' + time
    return time

class TrafficGenerator:
    #def __init__( self, param_path = 'params.json', model_path = 'model/model_direction' + str(1) + '.pt'):
    def __init__( self, param_path = 'params.json', model_path = 'model/model_direction1.pt'):        

        self.root_path = ''
        self.path_params = param_path
        self.model_path = model_path

        f = open(self.path_params)
        self.params = json.load(f)

        self.test_ratio = self.params["test_ratio"]
        self.sequence_length = self.params["sequence_length"]
        self.batch_size = self.params["batch_size"]
        self.layer_dim = self.params["layer_dim"]

        self.hidden_dim = self.params["hidden_dim"]
        self.dropout = self.params["dropout"]
        self.n_epochs = self.params["n_epochs"]
        self.learning_rate = self.params["learning_rate"]
        self.weight_decay = self.params["weight_decay"]
        self.optimizer_arg = self.params["optimizer"]
        self.activation_function = self.params["activation_function"]

        self.x_scaler = joblib.load('model/x_scaler.save')
        self.y_scaler = joblib.load('model/y_scaler.save')
        
        self.var = ['minute_sin','minute_cos','hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos','Year','air_pressure', 'moisture_percentage','rain_intensity','snow_depth','deg','visibility','wind_speed','accident_feature']

        self.input_dim = 17
        self.output_dim = 7
        self.model_params = {'input_dim': self.input_dim,
                        'hidden_dim' : self.hidden_dim,
                        'layer_dim' : self.layer_dim,
                        'output_dim' : self.output_dim,
                        'dropout_prob' : self.dropout,
                        'activation_function': self.activation_function}

        model = get_model('lstm', self.model_params)
        model.load_state_dict(torch.load(model_path))
    
        loss_fn = nn.MSELoss(reduction = "mean")
        if self.optimizer_arg == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        elif self.optimizer_arg == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = self.learning_rate, momentum = 0.9)

        self.opt = Optimization(model = model, loss_fn = loss_fn, optimizer = optimizer)
        
        self.seq_length = 8
        self.avg_weather = pd.read_csv('average_weather.csv')
        self.avg_weather['datetime'] = self.avg_weather.MONTH.astype(str) + '_' + self.avg_weather.DAY.astype(str) + '_' + self.avg_weather.time.astype(str)

    def generate_traffic(self, year, month, day, time):
        df = pd.DataFrame()
        df['Hour'] = [time] * 4 + [time + 1] * 4
        df['Time'] = df['Hour'].apply(lambda x: process_time(x)) + [i for i in [':00', ':15', ':30', ':45']] * 2
        df['Year'] = year
        df['MONTH'] = month
        df['DAY'] = day
        df['DATE'] = ''.join((str(year), '.', process_time(month), '.', process_time(day)))
        df['day_of_week'] = get_day_of_week(year, month, day)

        df['hour_sin'] =  list(map(lambda x: np.sin(2 * np.pi * (x) / 24), df.Hour))
        df['hour_cos'] =  list(map(lambda x: np.cos(2 * np.pi * (x)/ 24), df.Hour))
        df['day_sin'] =  list(map(lambda x: np.sin(2 * np.pi * (x - 1) / 7), df.day_of_week))
        df['day_cos'] =  list(map(lambda x: np.cos(2 * np.pi * (x - 1)/ 7), df.day_of_week))
        df['month_sin'] =  list(map(lambda x: np.sin(2 * np.pi * (x - 1) / 12), df.MONTH.astype(float)))
        df['month_cos'] =  list(map(lambda x: np.cos(2 * np.pi * (x - 1)/ 12), df.MONTH.astype(float)))
        df['Minute'] = list(map(lambda x: int(x.split(':')[1]), df.Time))
        df['minute_sin'] =  list(map(lambda x: np.sin(2 * np.pi * (x) / 60), df.Minute))
        df['minute_cos'] =  list(map(lambda x: np.sin(2 * np.pi * (x) / 60), df.Minute))

        other_features = ['air_pressure', 'moisture_percentage', 'rain_intensity', 'snow_depth', 'deg', 'visibility', 'wind_speed', 'accident_feature']
        
        df_x = df.iloc[0,:]
        df_str = ''.join((str(df_x.MONTH) + '_' + str(df_x.DAY) + '_' + df_x.Time))

        avg_w_idx = self.avg_weather[self.avg_weather.datetime == df_str].index[0]
        weather_df = self.avg_weather.loc[avg_w_idx:avg_w_idx + self.seq_length - 1]

        for v in other_features[:-1]:
            df[v] = weather_df[v].values
        df[other_features[-1]] = 0

        times = df.Time
        X = df[self.var]

        X_arr = self.x_scaler.transform(X)
        X_features = torch.Tensor(X_arr)

        preds = self.opt.predict(X_features)
        preds = self.y_scaler.inverse_transform(preds[:,0,:])
        generated_values = preprocess_generated_values(preds.copy())


        df = pd.DataFrame(generated_values)
        df.columns = ['HA_PA', 'KAIP', 'LinjAut', 'KAPP', 'KATP', 'HA_PK', 'HA_AV']
        df['Time'] = times

        return df



# Generating traffic for north-south direction
generator_dir1 = TrafficGenerator(model_path = 'model/model_direction1.pt')
# Generating traffic for south-north direction
generator_dir2 = TrafficGenerator(model_path = 'model/model_direction2.pt')

# Example: generating traffic for 22.1.2023 5pm-7pm
print(generator_dir1.generate_traffic(2030, 1, 22, 17))
print(generator_dir2.generate_traffic(2030, 1, 22, 17))
# Generator returns a pandas dataframe with the traffic flow numbers



# Example: generating traffic for the whole day hour by hour and then plotting the data
"""
year = 2030
month = 1
day = 22
li1 = []
li2 = []

for hour in range(22):
    df_dir1 = generator_dir1.generate_traffic(year, month, day, hour)
    li1.append(df_dir1.loc[1:4,:].copy())

    df_dir2 = generator_dir2.generate_traffic(year, month, day, hour)
    li2.append(df_dir2.loc[1:4,:].copy())


data_dir1 = pd.concat(li1, axis=0, ignore_index=True)
data_dir2 = pd.concat(li2, axis=0, ignore_index=True)

import matplotlib.pyplot as plt
plt.plot(data_dir1.HA_PA, label = 'Traffic in north-south direction')
plt.plot(data_dir2.HA_PA, label = 'Traffic in south-north direction')
plt.legend()
plt.show()
"""