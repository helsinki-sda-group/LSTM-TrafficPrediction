import argparse
import numpy as np
import pandas as pd
import torch

from torch import nn, optim
from sklearn.model_selection import train_test_split

import torch.optim as optim

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def power_law(x, beta):
    if x <= 0 or x >= 60:
        return 0
    else:
        return x**(-beta)

def feature_label_split(df, target_col):
    y = df[target_col]
    X = df.drop(columns = target_col)
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    if test_ratio == 0.0:
        X, y = feature_label_split(df, target_col)
        X_train = X
        y_train = y
        return X_train, y_train, 0, 0, 0, 0
    else:
        val_ratio = test_ratio / (1 - test_ratio)
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, shuffle = False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_ratio, shuffle = False)
        return X_train, X_val, X_test, y_train, y_val, y_test

# All predictions must be nonnegative integers
def postprocess_predictions(predictions):
    predictions[predictions < 0] = 0
    predictions = np.round(predictions)
    return predictions

def format_predictions(predictions, values, df_test, scaler, pred_vars, y_pred):
    vals = np.array(list(map(lambda x: scaler.inverse_transform(x), values)))
    preds = np.array(list(map(lambda x: scaler.inverse_transform(x), predictions)))

    vals = vals.reshape(vals.shape[0] * vals.shape[1], vals.shape[2])
    preds = preds.reshape(preds.shape[0] * preds.shape[1], preds.shape[2])

    preds = postprocess_predictions(preds)

    d = {}
    for i in range(len(y_pred)):
        d['value_' + pred_vars[i]] = vals[:, i]
        d['prediction_' + pred_vars[i]] = preds[:, i]

    df_result = pd.DataFrame(d , index = df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    return df_result

def calculate_metrics(df_result, pred_vars):
    metrics = {}
    for var in pred_vars:
        metrics['mae_' + var] = mean_absolute_error(df_result['value_' + var], df_result['prediction_' + var])
        metrics['mse_' + var] = mean_squared_error(df_result['value_' + var], df_result['prediction_' + var]) ** 0.5
        metrics['r2_' + var] = r2_score(df_result['value_' + var], df_result['prediction_' + var])
    metrics['mae_total'] = mean_absolute_error(df_result.iloc[:, len(pred_vars):], df_result.iloc[:, 0:len(pred_vars)])
    metrics['mse_total'] = mean_squared_error(df_result.iloc[:, len(pred_vars):], df_result.iloc[:, 0:len(pred_vars)]) ** 0.5
    metrics['r2_total'] = r2_score(df_result.iloc[:, len(pred_vars):], df_result.iloc[:, 0:len(pred_vars)])

    return metrics

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, activation_function):
        super(LSTMModel, self).__init__()

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

    def train_step(self, x, y):
        self.model.train()

        yhat = self.model(x)

        loss = self.loss_fn(y, yhat)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)

                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            print("[{}/{}] Training loss: {} \t Validation loss: {}".format(epoch, n_epochs, training_loss, validation_loss))


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


def train_model(args, data, var, y_pred, pred_vars, x_scaler, y_scaler, test_ratio = 0.2, sequence_length = 96, batch_size = 100, layer_dim = 3, hidden_dim = 512, dropout = 0.2, n_epochs = 100, learning_rate = 1e-3, weight_decay = 1e-6, optimizer_arg = 'Adam', activation_function = 'none'):
    data.fillna(0, inplace = True)

    df_features = data[var]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, pred_vars, test_ratio)

    X_train_arr = x_scaler.transform(X_train)
    X_val_arr = x_scaler.transform(X_val)
    X_test_arr = x_scaler.transform(X_test)

    y_train_arr = y_scaler.transform(y_train)
    y_val_arr = y_scaler.transform(y_val)
    y_test_arr = y_scaler.transform(y_test)

    train_features = torch.Tensor(X_train_arr)
    train_targets = torch.Tensor(y_train_arr)
    val_features = torch.Tensor(X_val_arr)
    val_targets = torch.Tensor(y_val_arr)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)


    b_size = (train_features.shape[0] -  train_features.shape[0] % sequence_length)
    train_features = train_features[:int(b_size)].view(int(b_size / sequence_length), sequence_length, train_features.shape[1])
    train_targets = train_targets[:int(b_size)].view(int(b_size / sequence_length), sequence_length, train_targets.shape[1])
    train = TensorDataset(train_features, train_targets)
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False, drop_last = True)

    b_size = (val_features.shape[0] -  val_features.shape[0] % sequence_length)
    val_features = val_features[:int(b_size)].view(int(b_size / sequence_length), sequence_length, val_features.shape[1])
    val_targets = val_targets[:int(b_size)].view(int(b_size / sequence_length), sequence_length, val_targets.shape[1])
    val = TensorDataset(val_features, val_targets)
    val_loader = DataLoader(val, batch_size = batch_size, shuffle = False, drop_last = True)

    b_size = (test_features.shape[0] -  test_features.shape[0] % sequence_length)
    test_features = test_features[:int(b_size)].view(int(b_size / sequence_length), sequence_length, test_features.shape[1])
    test_targets = test_targets[:int(b_size)].view(int(b_size / sequence_length), sequence_length, test_targets.shape[1])
    test = TensorDataset(test_features, test_targets)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = False, drop_last = True)
    test_loader_one = DataLoader(test, batch_size = 1, shuffle = False, drop_last = True)


    input_dim = len(X_train.columns)
    output_dim = len(y_pred)

    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout,
                    'activation_function': activation_function}


    model = get_model('lstm', model_params)

    loss_fn = nn.MSELoss(reduction = "mean")
    if optimizer_arg == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    elif optimizer_arg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

    opt = Optimization(model = model, loss_fn = loss_fn, optimizer = optimizer)

    print('Training started at {}'.format(datetime.now()))
    opt.train(train_loader, val_loader, batch_size = batch_size, n_epochs = n_epochs, n_features = input_dim)
    print('Training ended at {}'.format(datetime.now()))

    with open(args.loss_path + args.loss_name + '_train_losses.npy', 'wb') as f:
        np.save(f, opt.train_losses)
    with open(args.loss_path + args.loss_name + '_val_losses.npy', 'wb') as f:
        np.save(f, opt.val_losses)

    if args.save_model:
        print('Saving model to ' + args.model_path)
        torch.save(opt.model.state_dict(), args.model_path)

    preds, vals = opt.evaluate(test_loader_one, batch_size = 1, n_features = input_dim)
    predictions = np.array(list(map(lambda x: list(x[0]), preds)))
    values = np.array(list(map(lambda x: list(x[0]), vals)))

    df_result = format_predictions(predictions, values, X_test, y_scaler, pred_vars, y_pred)
    result_metrics = calculate_metrics(df_result, pred_vars)

    loss_dir = args.loss_path + args.loss_name + '_result_metrics.npy'
    print('saving result_metrics to ' + loss_dir)

    with open(args.loss_path + args.loss_name + '_result_metrics.npy', 'wb') as f:
        np.save(f, result_metrics)
    print(result_metrics)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--do_train',
        action = 'store_true'
    )
    argparser.add_argument(
        '--save_model',
        action = 'store_true'
    )
    argparser.add_argument(
        '--model_path',
       default = 'models/model.pt'
    )
    argparser.add_argument(
        '--data_path',
    )
    argparser.add_argument(
        '--params',
        default = 'params.json'
    )
    argparser.add_argument(
        '--loss_path',
        default = 'losses/'
    )
    argparser.add_argument(
        '--loss_name',
        default = 'exp'
    )
    argparser.add_argument(
        '--random_seed',
    )

    args = argparser.parse_args()

    if args.random_seed:
        import random
        random.seed(int(args.random_seed))
        torch.manual_seed(int(args.random_seed))
        np.random.seed(int(args.random_seed))

    import json

    f = open(args.params)
    params = json.load(f)
    f.close()

    test_ratio = params["test_ratio"]
    sequence_length = params["sequence_length"]
    batch_size = params["batch_size"]
    layer_dim = params["layer_dim"]
    hidden_dim = params["hidden_dim"]
    dropout = params["dropout"]
    n_epochs = params["n_epochs"]
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    optimizer_arg = params["optimizer"]
    activation_function = params["activation_function"]

    data = pd.read_csv(args.data_path)

    y_pred = [0, 1, 2, 3, 4, 5, 6]

    var = list(data.columns[6:])

    pred_vars = list(np.array(var)[y_pred])

    X, y = feature_label_split(data[var], pred_vars)
    x_scaler = MinMaxScaler()
    _ = x_scaler.fit_transform(X)
    y_scaler = MinMaxScaler()
    _ = y_scaler.fit_transform(y)

    if args.do_train:
        train_model(args, data, var, y_pred, pred_vars, x_scaler, y_scaler, test_ratio, sequence_length, batch_size, layer_dim, hidden_dim, dropout, n_epochs, learning_rate, weight_decay, optimizer_arg, activation_function)

if __name__ == '__main__':
    main()
