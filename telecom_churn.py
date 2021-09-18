import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore",)  #Ignore certain system-wide alerts
from time import time, strftime, gmtime
start = time()
import datetime
print(str(datetime.datetime.now()))

# Import Machine learning algorithm
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

#Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv(r'C:\Users\Beheerder\Desktop\Telecom_customer_churn.csv')
# Using the read.csv code of the pandas library, we read the Telecom_customer_churn.csv file and assign them to a variables called df.


print(df.head(3))
# Seeing first five row in df data

# from the data description, we can see that Customer_ID is unique - therefor it not provides us information we can learn.
df.drop(["Customer_ID"], axis = 1, inplace=True)

# We dropped the columns that seem to have no significant contribution to the model.
df.drop(['numbcars','dwllsize','HHstatin','ownrent','dwlltype','lor','income','adults','prizm_social_one','infobase','crclscod'],axis=1,inplace=True)

df['hnd_webcap']=df['hnd_webcap'].fillna('UNKW') # Handset web capability

df['avg6qty']=df['avg6qty'].fillna(df['avg6qty'].mean()) # Billing adjusted total number of calls over the life of the customer
df['avg6rev']=df['avg6rev'].fillna(df['avg6rev'].mean()) # Average monthly revenue over the life of the customer
df['avg6mou']=df['avg6mou'].fillna(df['avg6mou'].mean()) # Average monthly minutes of use over the life of the customer

df['change_mou']=df['change_mou'].fillna(df['change_mou'].mean()) # Percentage change in monthly minutes of use vs previous three month average
df['change_rev']=df['change_rev'].fillna(df['change_rev'].mean()) # Percentage change in monthly revenue vs previous three month average

df['rev_Mean']=df['rev_Mean'].fillna(df['rev_Mean'].mean())
df['totmrc_Mean']=df['totmrc_Mean'].fillna(df['totmrc_Mean'].mean())
df['da_Mean']=df['da_Mean'].fillna(df['da_Mean'].mean())
df['ovrmou_Mean']=df['ovrmou_Mean'].fillna(df['ovrmou_Mean'].mean())
df['ovrrev_Mean']=df['ovrrev_Mean'].fillna(df['ovrrev_Mean'].mean())
df['vceovr_Mean']=df['vceovr_Mean'].fillna(df['vceovr_Mean'].mean())
df['datovr_Mean']=df['datovr_Mean'].fillna(df['datovr_Mean'].mean())
df['roam_Mean']=df['roam_Mean'].fillna(df['roam_Mean'].mean())
df['mou_Mean']=df['mou_Mean'].fillna(df['mou_Mean'].mean())

df.dropna(inplace=True)

numerical_features = ['months', 'uniqsubs', 'actvsubs', 'totcalls', 'avg3qty', 'avg3rev', 'rev_Mean', 'mou_Mean',
                      'totmrc_Mean', 'da_Mean', 'ovrmou_Mean', 'datovr_Mean',
                      'roam_Mean', 'change_mou', 'change_rev', 'drop_vce_Mean', 'drop_dat_Mean', 'blck_vce_Mean',
                      'blck_dat_Mean', 'unan_vce_Mean', 'unan_dat_Mean',
                      'plcd_vce_Mean', 'plcd_dat_Mean', 'recv_vce_Mean', 'recv_sms_Mean', 'custcare_Mean',
                      'ccrndmou_Mean', 'threeway_Mean', 'mou_cvce_Mean',
                      'mou_cdat_Mean', 'mou_rvce_Mean', 'owylis_vce_Mean', 'mouowylisv_Mean', 'iwylis_vce_Mean',
                      'mouiwylisv_Mean', 'peak_vce_Mean', 'peak_dat_Mean',
                      'mou_peav_Mean', 'mou_pead_Mean', 'opk_vce_Mean', 'opk_dat_Mean', 'mou_opkv_Mean',
                      'drop_blk_Mean', 'callfwdv_Mean', 'callwait_Mean', 'totmou',
                      'totrev', 'avgrev', 'avgmou', 'avgqty', 'avg6mou', 'avg6rev', 'hnd_price', 'phones', 'models',
                      'truck', 'rv', 'forgntvl', 'eqpdays']

for i in numerical_features:
    f_sqrt = (lambda x: np.sqrt(abs(x)) if (x >= 1) or (x <= -1) else x)
    df[i] = df[i].apply(f_sqrt)


def detect_outliers(df, features):
    outlier_indices = []

    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c], 25)
        # 3rd quartile
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return outlier_indices

df.loc[detect_outliers(df,['uniqsubs', 'actvsubs'])]

# drop outliers
df = df.drop(detect_outliers(df,['uniqsubs', 'actvsubs']),axis = 0).reset_index(drop = True)

# Create correlation matrix
corr_matrix = df.corr().abs()
# print(corr_matrix)

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
df.drop(df[to_drop], axis=1, inplace=True)

# Unique variables of object columns
encoding_col=[]
for i in df.select_dtypes(include='object'):
    # print(i,'-->',df[i].nunique())
    encoding_col.append(i)

# one-hot encoding for variables with more than 2 categories
df2 = df.copy()
df2 = pd.get_dummies(df2, drop_first=True, columns = encoding_col, prefix = encoding_col)


# Import Machine learning algorithms
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

#Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix, ConfusionMatrixDisplay

#Split data into train and test sets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

# dependent and independent variables were determined.
X = df2.drop('churn', axis=1)
y = df2['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Defining the modelling function
def modeling(alg, alg_name, params={}):
    model = alg(**params)  # Instantiating the algorithm class and unpacking parameters if any
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance evaluation
    def print_scores(alg, y_true, y_pred):
        print(alg_name)
        acc_score = accuracy_score(y_true, y_pred)
        print("accuracy: ", acc_score)
        pre_score = precision_score(y_true, y_pred)
        print("precision: ", pre_score)
        rec_score = recall_score(y_true, y_pred)
        print("recall: ", rec_score)
        f_score = f1_score(y_true, y_pred, average='weighted')
        print("f1_score: ", f_score)

    print_scores(alg, y_test, y_pred)

    # cm = confusion_matrix(y_test, y_pred)
    # # Create the Confusion Matrix Display Object(cmd_obj).
    # cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['churn', 'notChurn'])
    #
    # # The plot() function has to be called for the sklearn visualization
    # cmd_obj.plot()
    #
    # # Use the Axes attribute 'ax_' to get to the underlying Axes object.
    # # The Axes object controls the labels for the X and the Y axes. It also controls the title.
    # cmd_obj.ax_.set(
    #     title='Sklearn Confusion Matrix with labels!!',
    #     xlabel='Predicted Churn',
    #     ylabel='Actual Churn')
    # # Finally, call the matplotlib show() function to display the visualization of the Confusion Matrix.
    # plt.show()

    return model

# LightGBM model
LGBM_model = modeling(lgb.LGBMClassifier, 'Light GBM')


#Saving best model
import joblib
#Sava the model to disk
filename = 'model2.sav'
joblib.dump(LGBM_model, filename)
