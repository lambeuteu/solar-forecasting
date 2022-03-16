#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transform_output_format import get_4D_output
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from utils import load_data_input
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
model_1 = LinearRegression()
X = np.random.random((10,5))
y = np.random.random((10,4))
# %%
model_1.fit(X,y)
# %%
GHI
# %%
seq = GHI[0]
seq_ravel = seq.reshape(4,81*81)
print(seq_ravel.shape)
# %%
plt.imshow(seq_ravel.reshape(4,81,81)[0])
# %%
def prepare_data(sequence):
    """_summary_

    Args:
        sequence (array(nb_examples,nb_img,81,81)): _description_
    """
    nb_samples, nb_img, size1, size2 = sequence.shape
    return sequence.reshape((nb_samples*size1*size2,nb_img))
# %%
model= LinearRegression()
X_train = prepare_data(GHI[:,:,15:66,15:66])
y_train_reshape = prepare_data(y_train)
# %%
model.fit(X_train,y_train_reshape)
# %%
model.coef_.shape
# %%
GHI_test,CLS_test,SZA_test,SAA_test,dates_test = load_data_input("X_test_copernicus.npz")
# %%
