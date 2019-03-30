import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import style
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import *


kc_df = pd.DataFrame()

kc_df = pd.read_csv('kc_house_data.csv')
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
feature_matrix = kc_df[features]
#feature_matrix = preprocessing.scale(feature_matrix)
feature_matrix_unscaled = kc_df[features]
lable_vector = kc_df['price']
feature_matrix_unscaled.head()
#feature_matrix[0::1000]

    
my_linear = LinearRegression()

my_linear.fit(feature_matrix, lable_vector)

# printing Theta0 using attribute "intercept_":
#print(my_linear.intercept_)

# printing [Theta1, Theta2, Theta3] using attribute "coef_":
#print(my_linear.coef_)
coef_list = my_linear.coef_
# coef_list.sort()
#print(coef_list)


sorted_features = ['grade','lat','sqft_living','sqft_above','waterfront','view']
best_feature_matrix = kc_df[sorted_features]
#scaling deature
#best_feature_matrix = preprocessing.scale(best_feature_matrix)

X_train, X_test, y_train, y_test = train_test_split(best_feature_matrix, lable_vector, test_size=0.3, random_state=3)
my_linear.fit(X_train, y_train)


ran_data = [345, 34, 678 , 145120, 12, 176451]
ran_data_arr = np.array(ran_data)
ran_data_num = ran_data_arr.reshape(1,-1)
y_predict_ln = my_linear.predict(ran_data_num)
print(y_predict_ln)

from sklearn.externals import joblib
joblib.dump(my_linear, "multiple.linear.pkl")


















