
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score,mean_squared_error 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



def draw_completion(df,head_size=100):
 
    data_nan = df.isna().sum().sort_values(ascending=False).head(head_size)
    plt.title('Proportion de NaN par variable (%)')
    sns.barplot(x=data_nan.values/df.shape[0]*100, y=data_nan.index)
## fonction renvoyant les colonnes et le pourcentage de valeurs nulles pour chacune d'elle
def columns_na_percentage(df):
    na_df=(df.isnull().sum()/len(df)*100).sort_values(ascending=False).reset_index()
    na_df.columns = ['Column','na_rate_percent']
    return na_df

## fonction qui trace la matrice de corrélation
def show_correlation_matrix(df,relevant_numeric_columns):
    corr_matrix = df[relevant_numeric_columns].corr()
    fig = plt.figure(1, figsize=(14, 14))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr_matrix, mask=mask, square=True, linewidths=0.1, annot=True)
    plt.xlim(0, corr_matrix.shape[1])
    plt.ylim(0, corr_matrix.shape[0])
    plt.show()
def corr_matrix(df,relevant_numeric_columns,threshold=0):
    corr_matrix = df[relevant_numeric_columns].corr().abs()
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                      .stack()
                      .sort_values(ascending=False))
    sol=sol[sol>=threshold]
    print(sol)

def linear_regression_func(df,target_col,feature_cols,test_size=0.25,random_state=42):
    coefs=dict()
    X=df[feature_cols].values
    y=df[target_col].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    linear_model=LinearRegression()
    linear_model.fit(X_train,y_train)
    y_pred=linear_model.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20,5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of "+target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of "+target_col)
    plt.legend(loc='best',fancybox=True, shadow=True)
    
    plt.show() 
   
    
    return linear_model.score(X_test,y_test),rmse,linear_model.coef_

def grid_search_cv_func(df,target_col,feature_cols,param_grid,scoring,model,test_size=0.25,random_state=42,cv=5):
    X_train,X_test,y_train,y_test=train_test_split(df[feature_cols].values,df[target_col].values,test_size=test_size,random_state=random_state)
    ridge= GridSearchCV(model, param_grid, scoring=scoring, cv=5)
    ridge.fit(X_train, y_train)
    y_pred=ridge.best_estimator_.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20,5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of "+target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of "+target_col)
    plt.legend(loc='best',fancybox=True, shadow=True)
    
    plt.show() 
    return ridge.best_params_,ridge.best_score_,rmse



def random_search_cv_func(df,target_col,feature_cols,param_grid,scoring,model,test_size=0.25,random_state=42,cv=5):
    X_train,X_test,y_train,y_test=train_test_split(df[feature_cols].values,df[target_col].values,test_size=test_size,random_state=random_state)
    ridge= RandomizedSearchCV(model, param_grid, scoring=scoring, cv=5)
    ridge.fit(X_train, y_train)
    y_pred=ridge.best_estimator_.predict(X_test)
    rmse= np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred))
    x_ax = range(len(y_test))
    plt.figure(figsize=(20,5))
    plt.plot(x_ax, y_test, linewidth=1, label="original values of "+target_col)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predictions of "+target_col)
    plt.legend(loc='best',fancybox=True, shadow=True)
    
    plt.show() 
    return ridge.best_params_,ridge.best_score_,rmse


def find_outliers(data,col,name):
  
    sorted_data=np.sort(data[col])
    Q3 = np.quantile(sorted_data, 0.75)
    Q1 = np.quantile(sorted_data, 0.25)
    IQR = Q3 - Q1   
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    outlier_free_list = [x for x in sorted_data if (
        (x < lower_range)| (x > upper_range))]

    outliers = data.loc[data[col].isin(outlier_free_list)]
     
    
    if len(outliers)>0:
        outliers=outliers[name].values.tolist()
        return  list(map(lambda x: (x,col), outliers))
    else:
        return []