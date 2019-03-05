from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#this need to be rewritten, 5 time stamps, but only last one contains the label
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all togethe
   
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values 
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#n_steps = 3
#output_features = 1
percent_split = 0.8


dataset = read_csv('./../resources/timeseries.csv', header=0, index_col=0)
print(dataset)
values = dataset.values
print(values[1,0])
values = values.astype('float32')
row, col = values.shape


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


reframed = series_to_supervised(scaled)

re_row, re_col = reframed.shape
print(re_col)
reframed.drop(reframed.columns[[i for i in range(re_col -1) if i >= col or i==36]], axis=1, inplace=True)
print(reframed.head())
value = reframed.values

split_point = int((reframed.shape[0])*percent_split)

train = value[:split_point, :]
test = value[split_point:, :]


onehotencoder = OneHotEncoder(categorical_features = [0])


train_X, train_y = train[:, :-1], onehotencoder.fit_transform(np.vstack(train[:, -1])).toarray()
test_X, test_y = test[:, :-1], onehotencoder.fit_transform(np.vstack(test[:, -1])).toarray()

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(2))
model.compile(loss='mae', optimizer='adam')
model.save('test.h5')

history = model.fit(train_X, train_y, epochs=10000, batch_size=50, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

'''
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
'''