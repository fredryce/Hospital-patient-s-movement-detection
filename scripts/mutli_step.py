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
from keras.layers import Dropout
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


def test():
	dataset = read_csv('./../resources/timeseries.csv', header=0, index_col=0)
	values = dataset.values
	values = values.astype('float32')
	row, col = values.shape

	reframed = series_to_supervised(values, 3).values
	with open('previous.txt', 'w') as f:
		f.write(str(reframed[2, 0:col])+'\n\n')
		f.write(str(reframed[2, col:2*col])+'\n\n')
		f.write(str(reframed[2, 2*col:3*col])+'\n\n')
		f.write(str(reframed[2, 3*col:])+'\n\n')



	#re_row, re_col = reframed.shape
	#print(re_col)
	#reframed.drop(reframed.columns[[i for i in range(re_col -1) if i >= features]], axis=1, inplace=True)


def main():
	n_steps = 3
	percent_split = 0.8


	dataset = read_csv('./../resources/timeseries.csv', header=0, index_col=0)
	print(dataset)
	values = dataset.values
	print(values[1,0])
	values = values.astype('float32')
	row, col = values.shape

	features = col * n_steps

	# normalize features
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)


	reframed = series_to_supervised(scaled, n_steps)

	re_row, re_col = reframed.shape

	drop_col = [i for i in range(re_col -1) if i >= features or i in [36,73, 110]]
	#print(drop_col)
	#print(reframed[reframed.columns[drop_col]])

	reframed.drop(reframed.columns[drop_col], axis=1, inplace=True)
	value = reframed.values

	#print(reframed)
	#exit()
	split_point = int((reframed.shape[0])*percent_split)

	train = value[:split_point, :]
	test = value[split_point:, :]


	onehotencoder = OneHotEncoder(categorical_features = [0])


	train_X, train_y = train[:, :-1], onehotencoder.fit_transform(np.vstack(train[:, -1])).toarray()
	test_X, test_y = test[:, :-1], onehotencoder.fit_transform(np.vstack(test[:, -1])).toarray()

	train_X = train_X.reshape((train_X.shape[0], n_steps, col-1))
	test_X = test_X.reshape((test_X.shape[0], n_steps, col-1))

	print(train_X.shape)

	model = Sequential()
	model.add(LSTM(50, return_sequences=False, input_shape=(train_X.shape[1], train_X.shape[2])))
	#model.add(Dropout(0.2))
	model.add(Dense(2, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	model.save('test.h5')

	history = model.fit(train_X, train_y, epochs=1000, batch_size=30, validation_data=(test_X, test_y), verbose=2, shuffle=False)
	# plot history
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()


if __name__ =="__main__":
	main()
	#test()