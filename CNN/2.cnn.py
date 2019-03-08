import pickle

from keras.optimizers import Adam, Nadam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout, SpatialDropout2D
from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow import set_random_seed
from keras.models import load_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import argparse
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import pdb


def create_model(optimizer=SGD,filters = 10, kernel_size=(3, 3), activation='tanh', dropout=0.4,loss='categorical_crossentropy',lr =0.01) :

	"""
	CREATE MODEL WITH MODEL PARAMETERS AS HYPERPARAMTERS TO TUNE
	"""

	model = Sequential([
	    Conv2D(filters = filters, kernel_size = (kernel_size,kernel_size) , activation = activation,
	           data_format = 'channels_last', input_shape = (34, 26, 29)),
	    MaxPooling2D(pool_size = (2,2)),
	    Dropout(rate = dropout),
	    Conv2D(filters = filters, kernel_size = (kernel_size,kernel_size), activation = activation),
	    Dropout(rate = dropout),
	    MaxPooling2D(pool_size = (2,2)),
	    Flatten(),
	    Dense(10, activation = activation),
	    Dense(2, activation ='softmax')
	])


	model.compile(optimizer=optimizer(lr=lr),
				  loss = loss,
				  metrics = ['accuracy'])

	return model


def eval_roc_auc(model): 
	"""
	INPUT : MODEL
	EVALUATE THE MODEL TO GET AUC, ACCURACY, ROC CURVE, FPR, TPR VALUES, ALSO SAVE ROC CURVE 
	"""


	preds = model.predict_proba(test_data)
	auc = roc_auc_score(test_classes, preds)
	acc = accuracy_score(test_classes[:,1], (preds[:,0]>preds[:,1])*1)


	print("Test accuracy: ", acc)

	print(f"Test AUROC: {auc}")

	fpr, tpr, thresholds = roc_curve(test_classes[:,1], preds[:,1])

	fig = plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	fig.savefig('../figures/ROC Curve.png')


#INSERTING COMMANDLINE OPTIONS FOR THIS SCRIPT
parser = argparse.ArgumentParser()
parser.add_argument("--TrainDataPath", help="Path to training data pkl",
                    nargs='?', default='/media/nehal/BLOO/NEHAL/UPENNACADS/SEM1/CISBE537/PROJECT/B3537_2018/B3537_2018/train_data.pkl', const=0)
parser.add_argument("--TestDataPath", help="Path to test data pkl",
                    nargs='?', default='/media/nehal/BLOO/NEHAL/UPENNACADS/SEM1/CISBE537/PROJECT/B3537_2018/B3537_2018/test_data.pkl', const=0)
args = parser.parse_args()



#LOADING TRAIN AND TEST DATASETS
with open(args.TrainDataPath, 'rb') as f:
    train_data, train_labels = pickle.load(f)
train_classes = to_categorical(train_labels)

with open(args.TestDataPath, 'rb') as f:
    test_data, test_labels = pickle.load(f)
test_classes = to_categorical(test_labels)


#HYPERPARAMETER COMBINATIONS
p = {'lr': [0.1, 0.01, 0.001],
     'filters':[8, 16, 32],
     'batch_size': [2, 16, 32],
     'epochs': [25,50,75],
     'dropout': [0, 0.2, 0.4],
     'optimizer': [SGD, Adam, Nadam, RMSprop],
     'loss': ['categorical_hinge','categorical_crossentropy'],
     'activation':['relu', 'elu', 'tanh'],
     'kernel_size' :[3,4]
     }


# p = {'lr': [0.1],
# 	'activation':['softmax'],
#      'kernel_size' :[(3,3)],
#      'filters':[8],
#           'epochs': [5]


#      }

seed = 7
np.random.seed(seed)

#KERAS CLASSIFIER TO USE IN SCIKIT WORKFLOW : Scikit-Learn classifier interface, 
#PREDICT FUNCTION OF SCIKIT AND KERAS WORKS DIFFERENTLY, USE predict_proba() for actual class probabilities
kerasmodel = KerasClassifier(build_fn=create_model)

#PERFORMING GRID SEARCH OVER THE HYPERPARAMETER GRID TO SEARCH FOR THE COMBINATION ON BEST ROC_AUC
grid = GridSearchCV(estimator = kerasmodel, param_grid = p, cv = 2, scoring= 'roc_auc',refit = True)


grid_result = grid.fit(train_data, train_classes,shuffle = 'true')
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

bestmodel = grid.best_estimator_

#EVALUATE THE BEST MODEL
eval_roc_auc(bestmodel)

bestmodel.save('../model/most_recent.h5')

# model = load_model('../model/most_recent.h5')



# # Check AUC on the training data, just to verify that the training data was learned.
# score = model.evaluate(train_data, train_classes)
# preds = model.predict(train_data)
# auc = roc_auc_score(train_classes, preds)
# print("Training data accuracy: ", score[1])
# print(f"Training AUROC: {auc}")
# print(len(train_classes))



