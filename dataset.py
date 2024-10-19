import pandas as pd
import numpy as np

def load_data(stock, dates, look_back):
  data_raw = stock.values
  dates = dates.values.tolist()
  data = []
  for index in range(len(data_raw)-look_back):
    data.append(data_raw[index: index+look_back])

  data = np.array(data)
  test_set_size = int(np.round(0.2*data.shape[0]))
  train_set_size = data.shape[0] - (test_set_size)

  x_train = data[:train_set_size, :-1,:]
  y_train = data[:train_set_size,-1,:]
  dates_train = dates[:train_set_size]

  x_test = data[train_set_size:,:-1]
  y_test = data[train_set_size:,-1,:]
  dates_test_plot = dates[train_set_size+look_back:]
  #if len(dates_test_plot) >= 2:
    # Change all elements in the middle to empty strings
    #dates_test_plot[1:-1] = [''] * (len(dates_test_plot) - 2)

  return x_train, y_train, x_test, y_test, dates_test_plot