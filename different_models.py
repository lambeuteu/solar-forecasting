#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transform_output_format import get_4D_output, get_2D_output
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from utils import load_data_input
import pickle
# %%
def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	models['lasso'] = Lasso()
	models['ridge'] = Ridge()
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	print('Defined %d models' % len(models))
	return models
 
# %%
models = get_models()
# %%
# fit a single model
def fit_model(model, X, y):
	# clone the model configuration
	local_model = clone(model)
	# fit the model
	local_model.fit(X, y)
	return local_model
 
# fit one model for each variable and each forecast lead time [var][time][model]
def fit_models(model, train):
	# prepare structure for saving models
	models = [[list() for _ in range(train.shape[1])] for _ in range(train.shape[0])]
	# enumerate vars
	for i in range(train.shape[0]):
		# enumerate lead times
		for j in range(train.shape[1]):
			# get data
			data = train[i, j]
			X, y = data[:, :-1], data[:, -1]
			# fit model
			local_model = fit_model(model, X, y)
			models[i][j].append(local_model)
	return models
# %%
GHI,CLS,SZA,SAA,dates = load_data_input("X_train_copernicus.npz")
# %%
y_train_csv = pd.read_csv('y_train_zRvpCeO_nQsYtKN.csv')
y_train = get_4D_output(y_train_csv)
# %%
def prepare_data(sequence):
    """_summary_

    Args:
        sequence (array(nb_examples,nb_img,81,81)): _description_
    """
    nb_samples, nb_img, size1, size2 = sequence.shape
    seq_swap = sequence.swapaxes(1,2).swapaxes(2,3)
    return seq_swap.reshape((nb_samples*size1*size2,nb_img))
# %%
model= LinearRegression()
#
GHI_train = prepare_data(GHI[:,:,15:66,15:66])
CLS_train = prepare_data(CLS[:,:,15:66,15:66])
X_train = np.concatenate([GHI_train, CLS_train],axis=1)
y_train_reshape = prepare_data(y_train)
# %%
model.fit(X_train,y_train_reshape)
# %%
model.coef_
model.intercept_
# %%
filename = 'linearreg_model.sav'
pickle.dump(model, open(filename, 'wb'))
# %%
filename = 'linearreg_model.sav'
model = pickle.load(open(filename, 'rb'))
# %%
GHI_test,CLS_test,SZA_test,SAA_test,dates_test = load_data_input("X_test_copernicus.npz")
# %%
GHI_test_r = prepare_data(GHI_test[:,:,15:66,15:66])
CLS_test_r = prepare_data(CLS_test[:,:,15:66,15:66])
X_test = np.concatenate([GHI_test_r, CLS_test_r], axis=1)
# %%
y_predict = model.predict(X_test)
# %%
y_preds = y_predict.reshape(1841,51,51,4)
y_preds = y_preds.swapaxes(2,3).swapaxes(1,2)
#%%
y_preds_2D = get_2D_output(y_preds)
y_preds_2D.to_csv('linearreg2.csv', index=False)
# %%
