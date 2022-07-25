
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




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