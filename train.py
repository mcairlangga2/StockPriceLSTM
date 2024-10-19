from dataset import load_data
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import math, time
import itertools
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from model import LSTM

def save_plot_data(df):
    plt.figure(figsize=(15,6))
    sns.lineplot(x = 'Date', y='Close', data=df)
    plt.title('Data Harga Saham PT Bank Syariah Indonesia Tbk')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Penutupan')
    plt.savefig('Data Saham BSI.png')

def preprocess(df, look_back):
    df_BSI = df.fillna(method='ffill')
    scaler = MinMaxScaler(feature_range=(-1,1))
    df_BSI['Close'] = scaler.fit_transform(df_BSI['Close'].values.reshape(-1,1))
    df_BSI = df_BSI[['Close']]
    x_train, y_train, x_test, y_test, dates_test = load_data(df_BSI, df['Date'],look_back)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    return x_train, x_test, y_train, y_test, scaler,dates_test

def prepare_for_plotting(df):
    jumlah_data = df.shape[0]
    print('jumlah data harga saham bank BSI {} hari'.format(jumlah_data))
    data_used = df[['Date','Close']]
    data_used['Date'] = pd.to_datetime(data_used['Date'])
    data_used.sort_values(by='Date', inplace=True)
    data_used.index = pd.RangeIndex(len(data_used.index))
    data_used.index = range(len(data_used.index))
    return data_used

def train_loop(model, optimiser, loss_fn, epoch):
    num_epochs = int(epoch)
    hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        #model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
    return model, hist, y_train_pred

def plot_train_loss(hist):
    plt.plot(hist, label="Training loss")
    plt.legend()
    plt.savefig()

def calculate_mape(y_true, y_pred):
  # Check for zeros in ground truth to avoid division by zero
  if np.any(y_true == 0):
    raise ValueError("MAPE cannot be calculated when ground truth values contain zeros.")

  # Calculate absolute percentage errors (APE)
  ape = np.abs((y_true - y_pred) / y_true)

  # Calculate MAPE as the mean of APEs
  mape = np.mean(ape)

  return mape

def test(model, x_test, y_test, y_train,y_train_pred,scaler):
    print('testing started')
    y_test_pred = model(x_test)
    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())
    scaler = MinMaxScaler(feature_range=(-1,1))
    # calculate root mean squared error
    train_mape = calculate_mape(y_train[:, 0], y_train_pred[:, 0])
    print('Train MAPE: %.4f' % train_mape)

    test_mape = calculate_mape(y_test[:, 0], y_test_pred[:, 0])
    print('Test MAPE: %.4f' % test_mape)
    return y_test_pred, test_mape, y_test

def plot_test(df, y_test, y_test_pred, sequence_length, layer, dimension, epoch, lr,dates_test):
    figure, axes = plt.subplots(figsize=(15, 6))
    #axes.xaxis_date()
    df_BSI = df.fillna(method='ffill')
    df_BSI = df_BSI[['Close']]
    axes.plot(dates_test, y_test, color = 'red', label = 'Harga Saham BSI Real')
    axes.plot(dates_test, y_test_pred, color = 'blue', label = 'Harga Saham BSI Hasil Prediksi')
    y_test_list = [price[0] for price in y_test]
    y_test_list_pred = [price_pred[0] for price_pred in y_test_pred]
    df_hasil = {'Tanggal':dates_test, 'Harga Saham Real':y_test_list, 'Harga Saham Prediksi':y_test_list_pred}
    df_hasil = pd.DataFrame(df_hasil)
    df_hasil.to_excel('Hasil prediksi model dengan {} sequence length, {} layer, {} hidden dimension, {} epoch, learning rate {}.xlsx'.format(sequence_length,layer,dimension,epoch,lr))
    dates_test_copy = dates_test.copy()
    dates_test_copy[1:-1] = [''] * (len(dates_test_copy) - 2)
    axes.set_xticklabels(dates_test_copy)
    plt.title('Prediksi Harga Saham BRI')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Saham')
    plt.legend()
    plt.savefig('Hasil prediksi model dengan {} sequence length, {} layer, {} hidden dimension, {} epoch, learning rate {}.png'.format(sequence_length,layer,dimension,epoch,lr))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epochs', nargs='+', required=True)
    parser.add_argument('--LSTM_layers', nargs='+', required=True)
    parser.add_argument('--sequence_length',nargs='+', required=True)
    parser.add_argument('--hidden_dim',  nargs='+', required=True)
    parser.add_argument('--lr', nargs='+', required=True )
    params = parser.parse_args()
    df = pd.read_csv(params.data_dir)
    data_used = prepare_for_plotting(df)
    save_plot_data(data_used)
    look_backs = params.sequence_length
    hidden_dims = params.hidden_dim
    layers = params.LSTM_layers
    epoch_list = params.epochs
    learning_rates = params.lr
    result_columns = ['Panjang Sekuens', 'Jumlah Layer', 'Jumlah Hidden Dimensions', 'Jumlah Epoch', 'Learning Rate', 'Test RMSE']
    result_df = pd.DataFrame(columns=result_columns)
    #print(result_df)
    for sequence_length in look_backs:
        x_train, x_test, y_train, y_test, scaler, dates_test = preprocess(df, int(sequence_length))
        for layer in layers:
            for dimensions in hidden_dims:
                for epoch in epoch_list:
                    for lr in learning_rates:
                        print('train the model using {} sequence length, {} layer, {} hidden dimension, {} epoch, with learning rate {}'.format(sequence_length, layer, dimensions, epoch, lr))
                        model = LSTM(input_dim=1, hidden_dim=int(dimensions), output_dim=1, num_layers=int(layer))
                        loss_fn = torch.nn.MSELoss()
                        optimiser = torch.optim.Adam(model.parameters(), lr=float(lr))
                        model_trained, hist, y_train_pred = train_loop(model, optimiser, loss_fn, epoch)
                        y_test_pred, testScore, y_test_out = test(model_trained, x_test, y_test, y_train,y_train_pred,scaler)
                        plot_test(df, y_test_out, y_test_pred, sequence_length, layer, dimensions, epoch, lr,dates_test)
                        insert_result = {'Panjang Sekuens':int(sequence_length), 'Jumlah Layer':int(layer), 'Jumlah Hidden Dimensions':int(dimensions), 'Jumlah Epoch':int(epoch), 'Learning Rate':float(lr), 'Test MAPE':testScore}
                        result_df = pd.concat([result_df, pd.DataFrame([insert_result])], ignore_index=True)
                        #print(result_df)
    result_df.to_csv('prediction_results.csv')
    print('experiment finished')
                    
                    


