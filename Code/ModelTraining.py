# Import the libraries
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
#import relativedelta
from dateutil.relativedelta import relativedelta
#import json
#import pyodbc
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense

from tf_keras.models import Model                      ### New
from tf_keras.layers import Input, LeakyReLU, Dense    ### New

from tensorflow.keras.layers import BatchNormalization
#from tensorflow.python.keras.layers import BatchNormalization

#Treand_Seasonality_Prepared = pd.read_csv('/content/TreandSeasonalityPrepared (5).csv')
Treand_Seasonality_Prepared = pd.read_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\TreandSeasonalityPrepared.csv')
Treand_Seasonality_Prepared['yrnmo'] = pd.to_datetime(Treand_Seasonality_Prepared['yrnmo'])
Treand_Seasonality_Prepared = Treand_Seasonality_Prepared.set_index('yrnmo')
# print(trendSeasonality_df)

# calculating trend
Treand_Final = pd.DataFrame(columns=["Year","Month","Actual","Trend","Portfolio"])

# Initialize an empty list to collect data
Treand_Final_list = []

groups_ts = Treand_Seasonality_Prepared.groupby(["Name"])
keys_ts = groups_ts.groups.keys()
for i in keys_ts:
    subsetData = groups_ts.get_group(i)
    subsetData_ = pd.DataFrame(subsetData)
    subsetData = pd.DataFrame(subsetData)
    subsetData = pd.DataFrame(subsetData[['count']])
    decomposition = seasonal_decompose(subsetData,model="additive")
    newData = pd.merge(subsetData,decomposition.trend,on="yrnmo")
    newData = newData.fillna(0.00)

    newData["Portfolio"] = subsetData_['Name'].iloc[0]
    #print(newData.dtypes)
    newData = newData.rename(columns={'count': 'Actual','trend': 'Trend'}, inplace=False)
    newData = newData.reset_index()
    newData['Year'] = pd.DatetimeIndex(newData['yrnmo']).year
    newData['Month'] = pd.DatetimeIndex(newData['yrnmo']).month
    newData = newData.drop(columns=['yrnmo'])
    # Add new data to the list
    Treand_Final_list.append(newData)

# Concatenate all data from the list into the final DataFrame
Treand_Final = pd.concat(Treand_Final_list, ignore_index=True)

Treand_Final.insert(0, 'RowNo', range(1, 1 + len(Treand_Final)))
Treand_Final['CreatedDate'] = datetime.now()
Treand_Final['UpdatedDate'] = datetime.now()
Treand_Final = Treand_Final[['RowNo', 'Year','Month','Actual','Trend','Portfolio','CreatedDate','UpdatedDate']]
print(Treand_Final)
print(Treand_Final.dtypes)


# calculating Seasonality
Seasonality_Final = pd.DataFrame(columns=["Month","Value","Portfolio"])

# Initialize an empty list to collect data
Seasonality_Final_list = []

groups_ts = Treand_Seasonality_Prepared.groupby(["Name"])
keys_ts = groups_ts.groups.keys()
for i in keys_ts:
    subsetData = groups_ts.get_group(i)
    subsetData_ = pd.DataFrame(subsetData)
    subsetData = pd.DataFrame(subsetData)
    subsetData = pd.DataFrame(subsetData[['count']])
    decomposition = seasonal_decompose(subsetData,model="additive")
    newData = pd.DataFrame(decomposition.seasonal)
    newData = newData.fillna(0.00)
    newData = newData.iloc[0:12]

    newData["Portfolio"] = subsetData_['Name'].iloc[0]
    newData = newData.rename(columns={'seasonal': 'Value'}, inplace=False)
    newData = newData.reset_index()
    newData['Month'] = pd.DatetimeIndex(newData['yrnmo']).month
    newData = newData.drop(columns=['yrnmo'])

    # Add new data to the list
    #Seasonality_Final = Seasonality_Final.append(newData)
    Seasonality_Final_list.append(newData)

# Concatenate all data from the list into the final DataFrame
Seasonality_Final = pd.concat(Seasonality_Final_list, ignore_index=True)

Seasonality_Final.insert(0, 'RowNo', range(1, 1 + len(Seasonality_Final)))
Seasonality_Final['CreatedDate'] = datetime.now()
Seasonality_Final['UpdatedDate'] = datetime.now()
Seasonality_Final = Seasonality_Final[['RowNo', 'Month','Value','Portfolio','CreatedDate','UpdatedDate']]
print(Seasonality_Final)
print(Seasonality_Final.dtypes)


# forecast trend for next 12 months

# Initialize empty list to collect data
Forecast_Final_list = []

# Group by Name
groups_ts = Treand_Seasonality_Prepared.groupby("Name")

for name, subsetData in groups_ts:
    subsetData_ = subsetData.copy()
    subsetData = subsetData[['count']]  # Keep only relevant column

    # Train-Test Split (Dynamic)
    split_index = int(len(subsetData) * 0.8)
    train, test = subsetData.iloc[:split_index], subsetData.iloc[split_index:]

    # ARIMA Model
    model_arima = auto_arima(train, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    model_arima.fit(train)
    forecast = model_arima.predict(n_periods=len(test))

    model_arima_all = auto_arima(subsetData, trace=True, error_action='ignore', suppress_warnings=True)
    forecast_new = model_arima_all.predict(n_periods=12)

    acc = np.sqrt(mean_squared_error(test['count'], forecast))

    # Simple Exponential Smoothing
    fit2 = SimpleExpSmoothing(train['count']).fit(smoothing_level=0.6, optimized=False)
    fit2_all = SimpleExpSmoothing(subsetData['count']).fit(smoothing_level=0.6, optimized=False)

    y_hat_avg = test.copy()
    y_hat_avg['SES'] = fit2.forecast(len(test))

    acc_expo = np.sqrt(mean_squared_error(test['count'], y_hat_avg['SES']))

    if acc > acc_expo:
        acc = acc_expo
        forecast_new = fit2_all.forecast(12)

    # Holt’s Linear Trend Method
    fit1 = Holt(train['count']).fit(smoothing_level=0.3, smoothing_slope=0.1)
    fit1_all = Holt(subsetData['count']).fit(smoothing_level=0.3, smoothing_slope=0.1)

    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    acc_holtslinear = np.sqrt(mean_squared_error(test['count'], y_hat_avg['Holt_linear']))

    if acc > acc_holtslinear:
        acc = acc_holtslinear
        forecast_new = fit1_all.forecast(12)

    # Holt-Winters Method
    if len(train) >= 12:
        fit3 = ExponentialSmoothing(train['count'], seasonal_periods=12, trend='add', seasonal='add').fit()
        fit3_all = ExponentialSmoothing(subsetData['count'], seasonal_periods=12, trend='add', seasonal='add').fit()
    else:
        fit3 = ExponentialSmoothing(train['count'], trend='add').fit()
        fit3_all = ExponentialSmoothing(subsetData['count'], trend='add').fit()

    y_hat_avg['Holt_Winter'] = fit3.forecast(len(test))
    acc_holtswinters = np.sqrt(mean_squared_error(test['count'], y_hat_avg['Holt_Winter']))

    if acc > acc_holtswinters:
        acc = acc_holtswinters
        forecast_new = fit3_all.forecast(12)

    # Date Handling
    last_date = pd.to_datetime(test.index[-1])
    date_range = pd.date_range(start=last_date, periods=13, freq='MS')[1:]

    # Creating Final DataFrame
    dataframe = pd.DataFrame({
        "Year": date_range.year,
        "Month": date_range.month,
        "Prediction": np.round(forecast_new).astype(int),
        "Portfolio": name
    })
    dataframe["Prediction"] = np.maximum(dataframe["Prediction"], 0)

    Forecast_Final_list.append(dataframe)

# Concatenate Results
Forecast_Final = pd.concat(Forecast_Final_list, ignore_index=True)
Forecast_Final.insert(0, 'RowNo', range(1, len(Forecast_Final) + 1))
Forecast_Final["CreatedDate"] = datetime.now()
Forecast_Final["UpdatedDate"] = datetime.now()
Forecast_Final = Forecast_Final[['RowNo', "Year","Month",'Portfolio',"Prediction",'CreatedDate','UpdatedDate']]
print(Forecast_Final)
print(Forecast_Final.dtypes)

# Save DataFrame to a CSV file
Treand_Final.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\interData\Treand_Final.csv', index=False)
Seasonality_Final.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\interData\Seasonality_Final.csv', index=False)
Forecast_Final.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\interData\Forecast_Final.csv', index=False)

FinalData_re_path_1 = pd.read_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\FinalDataRe.csv')
#FinalData_re_path_1 = pd.read_csv('/content/FinalDataRe_Copy.csv')
#FinalData_re_path_1 = FinalData_re_path1.drop('Column1',axis=1)  # What is Column1???
FinalData_re_path = FinalData_re_path_1.copy()
print(FinalData_re_path.columns)

# Convert categorical columns (excluding 'Deleted')
categorical_cols = [
    'TravelModeCode', 'ReligionCode', 'MaritalStatus', 'LivingCode',
    'EmployedCompanyCode', 'EmploymentTypeCode', 'GenderCode', 'ReportingPersonCode',
    'JoinedTypeCode', 'GradeCode', 'SpouseDisabled', 'SpouseAlive', 'SpouseOccupation'
]

for col in categorical_cols + ['Deleted']:  # Ensure 'Deleted' is still categorical
    FinalData_re_path[col] = FinalData_re_path[col].astype('category')

# Model training
Other_ = pd.DataFrame()
model_List = []
Accuraicies = []
findata_new = pd.DataFrame(columns=["Group.1","x","Portfolio"])

groups_ts = FinalData_re_path.groupby(["Name"])
keys_ts = groups_ts.groups.keys()
for i in keys_ts:
    subsetData = groups_ts.get_group(i)
    subsetData_ = pd.DataFrame(subsetData)
    subsetData = pd.DataFrame(subsetData)
    if(len(subsetData[subsetData['Deleted'] == 1]) >= 20) and (len(subsetData[subsetData['Deleted'] == 0]) >= 20):
        subsetData_port = subsetData
        subsetData = subsetData.drop(columns=["Name"])
        subsetData['Deleted'] = subsetData['Deleted'].astype('category')
        n = pd.get_dummies(subsetData, columns=["EmployedCompanyCode","EmploymentTypeCode","GenderCode","GradeCode","LivingCode","MaritalStatus","JoinedTypeCode","ReligionCode","ReportingPersonCode","TravelModeCode","SpouseAlive","SpouseDisabled","SpouseOccupation"])

        training_sample = n.sample(frac=0.8, random_state=66)
        test_Set_Considered = n[~n.index.isin(training_sample.index)]
        X = training_sample.drop('Deleted', axis=1)
        y = training_sample['Deleted']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

        # define encoder
        visible = Input(shape=(X_train.shape[1],))

        # encoder level 1
        e = Dense(X_train.shape[1] * 2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # encoder level 2
        e = Dense(X_train.shape[1])(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # bottleneck
        # n_bottleneck = round(float(X_train.shape[1]) / 2.0)
        # bottleneck = Dense(n_bottleneck)(e)

        # bottleneck
        n_bottleneck = X_train.shape[1]
        bottleneck = Dense(n_bottleneck)(e)

        # define decoder, level 1
        d = Dense(X_train.shape[1])(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)

        # decoder level 2
        d = Dense(X_train.shape[1] * 2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)

        # output layer
        output = Dense(X_train.shape[1], activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        # compile autoencoder model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # plot the autoencoder
        # fit the autoencoder model to reconstruct input
        
        # Convert to NumPy and ensure data type is float32
        X_train, X_test = X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32)

        # fit the autoencoder model to reconstruct input
        history = model.fit(X_train, X_train, epochs=100, batch_size=16, verbose=2, validation_data=(X_test,X_test))

        # define an encoder model (without the decoder)
        encoder = Model(inputs=visible, outputs=e)

        # encode the train data
        X_train_encode = encoder.predict(X_train)
        # encode the test data
        X_test_encode = encoder.predict(X_test)

        rfc = RandomForestClassifier(n_estimators=50, max_features=10,random_state=1)
        rfc.fit(X_train,y_train)
        rfc_predict = rfc.predict(X_test)

        rfc_cv_score = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')

        precision,recall,fscore,support = score(y_test, rfc_predict,average='macro')

        rf_Accuracy = accuracy_score(y_test, rfc_predict)

        confusion_matrix(y_test, rfc_predict)

        Random_Forest_Accuaracy = rf_Accuracy
        Random_Forest_Precision = precision
        Random_Forest_Recall = recall

        # Random_Forest_Summary = pd.DataFrame(columns=["Random_Forest_Accuaracy","Random_Forest_Precision","Random_Forest_Recall"],
        Random_Forest_Summary = pd.DataFrame(columns=["Random_Forest_Accuaracy",
                                                        "Random_Forest_Precision","Random_Forest_Recall"],
                                                data=[[Random_Forest_Accuaracy,Random_Forest_Precision,Random_Forest_Recall]])

        kappa = cohen_kappa_score(rfc_predict,y_test)

        matrix = confusion_matrix(y_test,rfc_predict, labels=[1,0])
        print('Confusion matrix : \n',matrix)

        feature_imp = pd.Series(rfc.feature_importances_,index=X.columns).sort_values(ascending=False)
        feature_imp = feature_imp.to_frame(name="Importance")
        feature_imp['Variable'] = feature_imp.index
        feature_imp['Variable'] = feature_imp['Variable'].str.split('_').str[0]
        feature_imp = feature_imp.reset_index(drop=True)
        feature_imp = feature_imp.groupby(['Variable'], sort=False)['Importance'].max()
        feature_imp = feature_imp.to_frame(name="Importance")
        feature_imp['Variable'] = feature_imp.index
        feature_imp["x"] = (feature_imp['Importance'] / sum(feature_imp['Importance'])) * 100
        feature_imp['Variable'] = feature_imp.index
        feature_imp = feature_imp.reset_index(drop=True)
        feature_imp = feature_imp.rename(columns={'Variable': 'Group.1'}, inplace=False)

        feature_imp['s'] = 1
        feature_imp['Kappa'] = kappa
        tab = feature_imp.iloc[0:10]
        tab = pd.DataFrame(tab[["s","Kappa","Group.1"]])
        tab
        s11 = pd.DataFrame(feature_imp[["Group.1","x"]])

        # Boosting

        # Gradient Boosting
        # from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        clf.fit(X_train, y_train)
        clf_predict = clf.predict(X_test)
        accuracy_score(y_test, clf_predict)

        # print("Accuracy:",metrics.accuracy_score(y_test, clf_predict))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, clf_predict))

        print("Classification Report")
        print(classification_report(y_test, clf_predict))

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(y_test, clf_predict,average='macro')

        # from sklearn.metrics import accuracy_score
        clf_Accuracy = accuracy_score(y_test, clf_predict)

        Boosting_Accuaracy = clf_Accuracy
        Boosting_Precision = precision
        Boosting_Recall = recall

        Boosting_Summary = pd.DataFrame(columns=["Boosting_Accuaracy","Boosting_Precision","Boosting_Recall"],
                                        data=[[Boosting_Accuaracy,Boosting_Precision,Boosting_Recall]])

        kappa_boost = cohen_kappa_score(clf_predict,y_test)

        s3_ = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
        s3_ = s3_.to_frame(name="Importance")
        s3_['Variable'] = s3_.index
        s3_['Variable'] = s3_['Variable'].str.split('_').str[0]
        s3_ = s3_.reset_index(drop=True)
        s3_ = s3_.groupby(['Variable'], sort=False)['Importance'].max()
        s3_ = s3_.to_frame(name="Importance")
        s3_['Variable'] = s3_.index
        s3_["x"] = (s3_['Importance'] / sum(s3_['Importance'])) * 100
        s3_['Variable'] = s3_.index
        s3_ = s3_.reset_index(drop=True)
        s3_ = s3_.rename(columns={'Variable': 'Group.1'}, inplace=False)

        s3_['s'] = 2
        s3_['Kappa'] = kappa_boost
        tab_1 = s3_ .iloc[0:10]
        tab_1 = pd.DataFrame(tab_1[["s","Kappa","Group.1"]])
        tab_1
        s33 = pd.DataFrame(s3_[["Group.1","x"]])

        # using Logistic regression
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
        params = {"C":np.logspace(-3,3,5,7), "penalty":["l1"]}
        model = LogisticRegression(penalty='l1', solver='liblinear')
        grid_search_cv = GridSearchCV(estimator=model,scoring='roc_auc', param_grid=params,cv=folds,return_train_score=True, verbose=1)
        modelnew = grid_search_cv.fit(X_train, y_train)
        # reviewing the results
        # cv_results = pd.DataFrame(grid_search_cv.cv_results_)
        y_pred = grid_search_cv.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report")
        print(classification_report(y_test, y_pred))

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(y_test, y_pred,average='macro')

        # from sklearn.metrics import accuracy_score
        Logistic_Accuracy = accuracy_score(y_test, y_pred)

        Logistic_Accuaracy = Logistic_Accuracy
        Logistic_Precision = precision
        Logistic_Recall = recall

        Logistic_Summary = pd.DataFrame(columns=["Logistic_Accuaracy","Logistic_Precision","Logistic_Recall"],
                                        data=[[Logistic_Accuaracy,Logistic_Precision,Logistic_Recall]])

        kappa_Logistic = cohen_kappa_score(y_pred,y_test)

        arr = modelnew.best_estimator_.coef_
        newarr = arr.reshape(len(modelnew.best_estimator_.coef_[0]),)
        z = pd.Series(newarr,index=X.columns).sort_values(ascending=False)
        z = z.to_frame(name="Importance")
        z['Variable'] = z.index
        z['Variable'] = z['Variable'].str.split('_').str[0]
        z = z.reset_index(drop=True)
        z['Importance'] = z['Importance'].abs()

        z['Importance'] = z['Importance'] + abs(min(z['Importance'])) + 1
        z = z.sort_values('Importance',ascending=False)
        z = z.groupby(['Variable'], sort=False)['Importance'].max()
        z = z.to_frame(name="Importance")
        z['Variable'] = z.index
        z["x"] = (z['Importance'] / sum(z['Importance'])) * 100
        z['Variable'] = z.index
        z = z.reset_index(drop=True)
        z = z.rename(columns={'Variable': 'Group.1'}, inplace=False)

        z['s'] = 3
        z['Kappa'] = kappa_Logistic
        w1 = z.iloc[0:10]
        w1 = pd.DataFrame(w1[["s","Kappa","Group.1"]])
        w1
        zz = pd.DataFrame(z[["Group.1","x"]])

        # ranking based on the results
        tab0 = pd.concat([tab,tab_1,w1])
        # tab=tab0
        # data = [[Random_Forest_Accuaracy,Random_Forest_Precision,Random_Forest_Recall], [Boosting_Accuaracy,Boosting_Precision,Boosting_Recall],[Logistic_Accuaracy,Logistic_Precision,Logistic_Recall]]
        data = [[Random_Forest_Accuaracy,Random_Forest_Precision,Random_Forest_Recall], [Boosting_Accuaracy,Boosting_Precision,Boosting_Recall], [Logistic_Accuaracy,Logistic_Precision,Logistic_Recall]]
        columns = ["Accuracy","Precision","Recall"]
        dataframe_new_check_original = pd.DataFrame(data,columns=columns)

        dataframe_new_check_original['rank_Accuracy_original'] = dataframe_new_check_original['Accuracy'].rank(na_option='top')
        dataframe_new_check_original['rank_Precision_original'] = dataframe_new_check_original['Precision'].rank(na_option='top')
        dataframe_new_check_original['rank_Recall_original'] = dataframe_new_check_original['Recall'].rank(na_option='top')
        dataframe_new_final_original = pd.DataFrame(dataframe_new_check_original[["rank_Accuracy_original","rank_Precision_original","rank_Recall_original"]])
        dataframe_new_final_original['mean'] = dataframe_new_final_original.mean(axis=1)

        dataframe_new_final_original['Overall_Rank'] = dataframe_new_final_original['mean'].rank(na_option='top')

        dataframe_new_final_original.rename(columns={'rank_Accuracy_original': 'rankAccuracy', 'rank_Precision_original': 'rankPrecision',
                                                        'rank_Recall_original': 'rankRecall'})

        # Obtaining the strengths of the results with completely new dataset#######
        # Obtaining the strengths of the results with completely new dataset
        # with Random forest

        # test_Set_Considered = n[~n.index.isin(training_sample.index)]
        test_Set_Considered_true = test_Set_Considered['Deleted']
        test_Set_Considered = test_Set_Considered.drop('Deleted', axis=1)
        pred4_newdataset = rfc.predict(test_Set_Considered)

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(test_Set_Considered_true, pred4_newdataset,average='macro')

        # from sklearn.metrics import accuracy_score
        rf_Accuracy = accuracy_score(test_Set_Considered_true, pred4_newdataset)

        confusion_matrix(test_Set_Considered_true, pred4_newdataset)

        Random_Forest_Accuaracy_new = rf_Accuracy
        Random_Forest_Precision_new = precision
        Random_Forest_Recall_new = recall

        # Random_Forest_Summary_newdataset = pd.DataFrame(columns=["Random_Forest_Accuaracy","Random_Forest_Precision","Random_Forest_Recall"],
        Random_Forest_Summary_newdataset = pd.DataFrame(columns=["Random_Forest_Accuaracy",
                                                                    "Random_Forest_Precision","Random_Forest_Recall"],
                                                        data=[[Random_Forest_Accuaracy_new,Random_Forest_Precision_new,Random_Forest_Recall_new]])

        # Boosting
        pred3_newdataset = clf.predict(test_Set_Considered)

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(test_Set_Considered_true, pred3_newdataset,average='macro')

        # from sklearn.metrics import accuracy_score
        clf_Accuracy = accuracy_score(test_Set_Considered_true, pred3_newdataset)

        confusion_matrix(test_Set_Considered_true, pred3_newdataset)

        Boosting_Accuaracy_new = clf_Accuracy
        Boosting_Precision_new = precision
        Boosting_Recall_new = recall

        Boosting_Summary_newdataset = pd.DataFrame(columns=["Boosting_Accuaracy","Boosting_Precision","Boosting_Recall"],
                                                    data=[[Boosting_Accuaracy_new,Boosting_Precision_new,Boosting_Recall_new]])

        # logistic regression
        probabilities_new = grid_search_cv.predict(test_Set_Considered)

        precision,recall,fscore,support = score(test_Set_Considered_true, probabilities_new,average='macro')

        log_Accuracy = accuracy_score(test_Set_Considered_true, probabilities_new)

        a = confusion_matrix(test_Set_Considered_true, probabilities_new)

        Logistic_Accuaracy_new = log_Accuracy
        Logistic_Precision_new = precision
        Logistic_Recall_new = recall

        Logistic_Summary_newdataset = pd.DataFrame(columns=["Logistic_Accuaracy",
                                                            "Logistic_Precision","Logistic_Recall"],
                                                    data=[[Logistic_Accuaracy_new,Logistic_Precision_new,Logistic_Recall_new]])

        # Ranking based on new dataset
        data = [[Random_Forest_Accuaracy_new,Random_Forest_Precision_new,Random_Forest_Recall_new],
                [Boosting_Accuaracy_new,Boosting_Precision_new,Boosting_Recall_new],
                [Logistic_Accuaracy_new,Logistic_Precision_new,Logistic_Recall_new]]
        columns = ["Accuracy","Precision","Recall"]
        dataframe_new_check = pd.DataFrame(data, columns=columns)

        dataframe_new_check['rank_Accuracy'] = dataframe_new_check['Accuracy'].rank(na_option='top')
        dataframe_new_check['rank_Precision'] = dataframe_new_check['Precision'].rank(na_option='top')
        dataframe_new_check['rank_Recall'] = dataframe_new_check['Recall'].rank(na_option='top')
        dataframe_new_final = pd.DataFrame(dataframe_new_check[["rank_Accuracy","rank_Precision",
                                                                "rank_Recall"]])
        dataframe_new_final['mean'] = dataframe_new_final.mean(axis=1)

        dataframe_new_final['Overall_Rank'] = dataframe_new_final['mean'].rank(na_option='top')

        dataframe_new_final.rename(columns={'rank_Accuracy': 'rankAccuracy', 'rank_Precision': 'rankPrecision',
                                            'rank_Recall': 'rankRecall'})

        tab['s'] = dataframe_new_final.iloc[0][4]
        tab['Accuracy'] = Random_Forest_Accuaracy
        tab_1['s'] = dataframe_new_final.iloc[1][4]
        tab_1['Accuracy'] = Boosting_Accuaracy
        w1['s'] = dataframe_new_final.iloc[2][4]
        w1['Accuracy'] = Logistic_Accuaracy

        tab = pd.concat([tab, tab_1,w1])
        tab = pd.concat([tab,tab_1,w1])
        tab0 = tab.sort_values("s", ascending=False)
        imp = tab0.drop(tab0.columns[[0, 1, 3]], axis=1)

        model_used = []
        model_accuracy = []

        # to select the model
        if (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[1][4]) & (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[2][4]):
            findata = s11
            model_used = rfc
            model_accuracy = Random_Forest_Accuaracy
        elif (dataframe_new_final.iloc[1][4] > dataframe_new_final.iloc[0][4]) & (dataframe_new_final.iloc[1][4] > dataframe_new_final.iloc[2][4]):
            findata = s33
            model_used = clf
            model_accuracy = Boosting_Accuaracy
        elif (dataframe_new_final.iloc[2][4] > dataframe_new_final.iloc[0][4]) & (dataframe_new_final.iloc[2][4] > dataframe_new_final.iloc[1][4]):
            findata = zz
            model_used = modelnew
            model_accuracy = Logistic_Accuaracy

        if (dataframe_new_final.iloc[0][4] == dataframe_new_final.iloc[1][4]) & (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[2][4]):
            if(dataframe_new_final.iloc[0][0] > dataframe_new_final.iloc[1][0]):
                findata = s11
                model_used = rfc
                model_accuracy = Random_Forest_Accuaracy
            elif(dataframe_new_final.iloc[0][0] < dataframe_new_final.iloc[1][0]):
                findata = s33
                model_used = clf
                model_accuracy = Boosting_Accuaracy
            elif(dataframe_new_final.iloc[0][0] == dataframe_new_final.iloc[1][0]):
                if(dataframe_new_final.iloc[0][1] > dataframe_new_final.iloc[1][1]):
                    findata = s11
                    model_used = rfc
                    model_accuracy = Random_Forest_Accuaracy
                elif(dataframe_new_final.iloc[0][1] < dataframe_new_final.iloc[1][1]):
                    findata = s33
                    model_used = clf
                    model_accuracy = Boosting_Accuaracy
                elif(dataframe_new_final.iloc[0][1] == dataframe_new_final.iloc[1][1]):
                    if(dataframe_new_final.iloc[0][2] > dataframe_new_final.iloc[1][2]):
                        findata = s11
                        model_used = rfc
                        model_accuracy = Random_Forest_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] < dataframe_new_final.iloc[1][2]):
                        findata = s33
                        model_used = clf
                        model_accuracy = Boosting_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] == dataframe_new_final.iloc[1][2]):
                        if(dataframe_new_final_original.iloc[0][4] > dataframe_new_final_original.iloc[1][4]):
                            findata = s11
                            model_used = rfc
                            model_accuracy = Random_Forest_Accuaracy
                        elif(dataframe_new_final_original.iloc[0][4] < dataframe_new_final_original.iloc[1][4]):
                            findata = s33
                            model_used = clf
                            model_accuracy = Boosting_Accuaracy

        if (dataframe_new_final.iloc[0][4] == dataframe_new_final.iloc[2][4]) & (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[1][4]):
            if(dataframe_new_final.iloc[0][0] > dataframe_new_final.iloc[2][0]):
                findata = s11
                model_used = rfc
                model_accuracy = Random_Forest_Accuaracy
            elif(dataframe_new_final.iloc[0][0] < dataframe_new_final.iloc[2][0]):
                findata = zz
                model_used = modelnew
                model_accuracy = Logistic_Accuaracy
            elif(dataframe_new_final.iloc[0][0] == dataframe_new_final.iloc[2][0]):
                if(dataframe_new_final.iloc[0][1] > dataframe_new_final.iloc[2][1]):
                    findata = s11
                    model_used = rfc
                    model_accuracy = Random_Forest_Accuaracy
                elif(dataframe_new_final.iloc[0][1] < dataframe_new_final.iloc[2][1]):
                    findata = zz
                    model_used = modelnew
                    model_accuracy = Logistic_Accuaracy
                elif(dataframe_new_final.iloc[0][1] == dataframe_new_final.iloc[2][1]):
                    if(dataframe_new_final.iloc[0][2] > dataframe_new_final.iloc[2][2]):
                        findata = s11
                        model_used = rfc
                        model_accuracy = Random_Forest_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] < dataframe_new_final.iloc[2][2]):
                        findata = zz
                        model_used = modelnew
                        model_accuracy = Logistic_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] == dataframe_new_final.iloc[2][2]):
                        if(dataframe_new_final_original.iloc[0][4] > dataframe_new_final_original.iloc[2][4]):
                            findata = s11
                            model_used = rfc
                            model_accuracy = Random_Forest_Accuaracy
                        elif(dataframe_new_final_original.iloc[0][4] < dataframe_new_final_original.iloc[2][4]):
                            findata = zz
                            model_used = modelnew
                            model_accuracy = Logistic_Accuaracy
            # findata=findata.loc[findata['x'] >0]
            # findata['Portfolio']=subsetData_port['Name']

        if (dataframe_new_final.iloc[1][4] == dataframe_new_final.iloc[2][4]) & (dataframe_new_final.iloc[1][4] > dataframe_new_final.iloc[0][4]):
            if(dataframe_new_final.iloc[1][0] > dataframe_new_final.iloc[2][0]):
                findata = s33
                model_used = clf
                model_accuracy = Boosting_Accuaracy
            elif(dataframe_new_final.iloc[1][0] < dataframe_new_final.iloc[2][0]):
                findata = zz
                model_used = modelnew
                model_accuracy = Logistic_Accuaracy
            elif(dataframe_new_final.iloc[1][0] == dataframe_new_final.iloc[2][0]):
                if(dataframe_new_final.iloc[1][1] > dataframe_new_final.iloc[2][1]):
                    findata = s33
                    model_used = clf
                    model_accuracy = Boosting_Accuaracy
                elif(dataframe_new_final.iloc[1][1] < dataframe_new_final.iloc[2][1]):
                    findata = zz
                    model_used = modelnew
                    model_accuracy = Logistic_Accuaracy
                elif(dataframe_new_final.iloc[1][1] == dataframe_new_final.iloc[2][1]):
                    if(dataframe_new_final.iloc[1][2] > dataframe_new_final.iloc[2][2]):
                        findata = s33
                        model_used = clf
                        model_accuracy = Boosting_Accuaracy
                    elif(dataframe_new_final.iloc[1][2] < dataframe_new_final.iloc[2][2]):
                        findata = zz
                        model_used = modelnew
                        model_accuracy = Logistic_Accuaracy
                    elif(dataframe_new_final.iloc[1][2] == dataframe_new_final.iloc[2][2]):
                        if(dataframe_new_final_original.iloc[1][4] > dataframe_new_final_original.iloc[2][4]):
                            findata = s33
                            model_used = clf
                            model_accuracy = Boosting_Accuaracy
                        elif(dataframe_new_final_original.iloc[1][4] < dataframe_new_final_original.iloc[2][4]):
                            findata = zz
                            model_used = modelnew
                            model_accuracy = Logistic_Accuaracy

            # findata=findata.loc[findata['x'] >0]
            # findata['Portfolio']=subsetData_port['Name']

        if(dataframe_new_final.iloc[0][4] == dataframe_new_final.iloc[1][4] == dataframe_new_final.iloc[2][4]):
            findata = s11
            model_used = rfc
            model_accuracy = Random_Forest_Accuaracy

        if model_used == []:
            model_used = rfc
            model_accuracy = Random_Forest_Accuaracy

        feature_list = findata['Group.1'].values
        data2 = pd.DataFrame()
        for u in range(len(feature_list)):
            data1 = pd.DataFrame(columns=['Corr'])
            # mylist =[]
            if (FinalData_re_path.columns.isin(feature_list).any()):
                corr = FinalData_re_path[feature_list[u]].corr(FinalData_re_path['Deleted'])
                a = [corr]
                df = pd.DataFrame(a,columns=["Correlation"])
                data2 = pd.concat([data2, df], ignore_index=True)

        findata["Correlation"] = data2["Correlation"]
        findata["Correlation"] = findata["Correlation"].fillna(0)
        findata = findata.loc[findata['x'] > 0]
        findata['Portfolio'] = subsetData_port['Name'].iloc[0]
        findata_new = pd.concat([findata_new, findata], ignore_index=True)

        # findata_new=findata_new.append(findata)
        model_List.append(model_used)
        Accuraicies.append(model_accuracy)

    else:
        Other_ = Other_.append(subsetData)

subsetData = Other_

findata = pd.DataFrame(columns=["Group.1","x"])
# model_used=[]
if not subsetData.empty:

    if(len(subsetData[subsetData['Deleted'] == 1]) >= 20) and (len(subsetData[subsetData['Deleted'] == 0]) >= 20):
        subsetData = subsetData.drop(columns=["Name"])
        subsetData['Deleted'] = subsetData['Deleted'].astype('category')
        n = pd.get_dummies(subsetData, columns=["EmployedCompanyCode","EmploymentTypeCode","GenderCode","GradeCode","LivingCode","MaritalStatus","JoinedTypeCode","ReligionCode","ReportingPersonCode","TravelModeCode","SpouseAlive","SpouseDisabled","SpouseOccupation"])

        training_sample = n.sample(frac=0.8, random_state=66)
        test_Set_Considered = n[~n.index.isin(training_sample.index)]
        X = training_sample.drop('Deleted', axis=1)
        y = training_sample['Deleted']

        # from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

        # define encoder
        visible = Input(shape=(X_train.shape[1],))

        # encoder level 1
        e = Dense(X_train.shape[1] * 2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # encoder level 2
        e = Dense(X_train.shape[1])(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # bottleneck
        # n_bottleneck = round(float(X_train.shape[1]) / 2.0)
        # bottleneck = Dense(n_bottleneck)(e)

        # bottleneck
        n_bottleneck = X_train.shape[1]
        bottleneck = Dense(n_bottleneck)(e)

        # define decoder, level 1
        d = Dense(X_train.shape[1])(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)

        # decoder level 2
        d = Dense(X_train.shape[1] * 2)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)

        # output layer
        output = Dense(X_train.shape[1], activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        # compile autoencoder model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # plot the autoencoder
        # fit the autoencoder model to reconstruct input

        # Convert to NumPy and ensure data type is float32
        X_train, X_test = X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32)

        # fit the autoencoder model to reconstruct input
        history = model.fit(X_train, X_train, epochs=100, batch_size=16, verbose=2, validation_data=(X_test, X_test))

        # define an encoder model (without the decoder)
        encoder = Model(inputs=visible, outputs=e)

        # encode the train data
        X_train_encode = encoder.predict(X_train)
        # encode the test data
        X_test_encode = encoder.predict(X_test)

        rfc = RandomForestClassifier(n_estimators=50, max_features=10,random_state=1)
        rfc.fit(X_train,y_train)
        rfc_predict = rfc.predict(X_test)

        rfc_cv_score = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')

        precision,recall,fscore,support = score(y_test, rfc_predict,average='macro')

        rf_Accuracy = accuracy_score(y_test, rfc_predict)

        confusion_matrix(y_test, rfc_predict)

        Random_Forest_Accuaracy = rf_Accuracy
        Random_Forest_Precision = precision
        Random_Forest_Recall = recall

        # Random_Forest_Summary = pd.DataFrame(columns=["Random_Forest_Accuaracy","Random_Forest_Precision","Random_Forest_Recall"],
        Random_Forest_Summary = pd.DataFrame(columns=["Random_Forest_Accuaracy",
                                                        "Random_Forest_Precision","Random_Forest_Recall"],
                                                data=[[Random_Forest_Accuaracy,Random_Forest_Precision,Random_Forest_Recall]])

        kappa = cohen_kappa_score(rfc_predict,y_test)

        matrix = confusion_matrix(y_test,rfc_predict, labels=[1,0])
        print('Confusion matrix : \n',matrix)

        feature_imp = pd.Series(rfc.feature_importances_,index=X.columns).sort_values(ascending=False)
        feature_imp = feature_imp.to_frame(name="Importance")
        feature_imp['Variable'] = feature_imp.index
        feature_imp['Variable'] = feature_imp['Variable'].str.split('_').str[0]
        feature_imp = feature_imp.reset_index(drop=True)
        feature_imp = feature_imp.groupby(['Variable'], sort=False)['Importance'].max()
        feature_imp = feature_imp.to_frame(name="Importance")
        feature_imp['Variable'] = feature_imp.index
        feature_imp["x"] = (feature_imp['Importance'] / sum(feature_imp['Importance'])) * 100
        feature_imp['Variable'] = feature_imp.index
        feature_imp = feature_imp.reset_index(drop=True)
        feature_imp = feature_imp.rename(columns={'Variable': 'Group.1'}, inplace=False)

        feature_imp['s'] = 1
        feature_imp['Kappa'] = kappa
        tab = feature_imp.iloc[0:10]
        tab = pd.DataFrame(tab[["s","Kappa","Group.1"]])
        tab
        s11 = pd.DataFrame(feature_imp[["Group.1","x"]])

        # Boosting

        # Gradient Boosting
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        clf.fit(X_train, y_train)
        clf_predict = clf.predict(X_test)
        accuracy_score(y_test, clf_predict)

        # print("Accuracy:",metrics.accuracy_score(y_test, clf_predict))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, clf_predict))

        print("Classification Report")
        print(classification_report(y_test, clf_predict))

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(y_test, clf_predict,average='macro')

        # from sklearn.metrics import accuracy_score
        precision,recall,fscore,support = score(y_test, clf_predict,average='macro')

        clf_Accuracy = accuracy_score(y_test, clf_predict)

        Boosting_Accuaracy = clf_Accuracy
        Boosting_Precision = precision
        Boosting_Recall = recall

        Boosting_Summary = pd.DataFrame(columns=["Boosting_Accuaracy",
                                                    "Boosting_Precision","Boosting_Recall"],
                                        data=[[Boosting_Accuaracy,Boosting_Precision,Boosting_Recall]])

        # from sklearn.metrics import cohen_kappa_score
        kappa_boost = cohen_kappa_score(clf_predict,y_test)

        s3_ = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
        s3_ = s3_.to_frame(name="Importance")
        s3_['Variable'] = s3_.index
        s3_['Variable'] = s3_['Variable'].str.split('_').str[0]
        s3_ = s3_.reset_index(drop=True)
        s3_ = s3_.groupby(['Variable'], sort=False)['Importance'].max()
        s3_ = s3_.to_frame(name="Importance")
        s3_['Variable'] = s3_.index
        s3_["x"] = (s3_['Importance'] / sum(s3_['Importance'])) * 100
        s3_['Variable'] = s3_.index
        s3_ = s3_.reset_index(drop=True)
        s3_ = s3_.rename(columns={'Variable': 'Group.1'}, inplace=False)

        s3_['s'] = 2
        s3_['Kappa'] = kappa_boost
        tab_1 = s3_ .iloc[0:10]
        tab_1 = pd.DataFrame(tab_1[["s","Kappa","Group.1"]])
        tab_1
        s33 = pd.DataFrame(s3_[["Group.1","x"]])

        # using Logistic regression
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
        params = {"C":np.logspace(-3,3,5,7), "penalty":["l1"]}
        model = LogisticRegression(penalty='l1', solver='liblinear')
        grid_search_cv = GridSearchCV(estimator=model,scoring='roc_auc', param_grid=params,cv=folds,return_train_score=True, verbose=1)
        grid_search_cv = GridSearchCV(estimator=model,
                                        scoring='roc_auc', param_grid=params,
                                        cv=folds,
                                        return_train_score=True, verbose=1)
        modelnew = grid_search_cv.fit(X_train, y_train)
        # reviewing the results
        # cv_results = pd.DataFrame(grid_search_cv.cv_results_)
        y_pred = grid_search_cv.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report")
        print(classification_report(y_test, y_pred))

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(y_test, y_pred,average='macro')

        # from sklearn.metrics import accuracy_score
        precision,recall,fscore,support = score(y_test, y_pred,average='macro')

        Logistic_Accuracy = accuracy_score(y_test, y_pred)

        Logistic_Accuaracy = Logistic_Accuracy
        Logistic_Precision = precision
        Logistic_Recall = recall

        Logistic_Summary = pd.DataFrame(columns=["Logistic_Accuaracy",
                                                    "Logistic_Precision","Logistic_Recall"],
                                        data=[[Logistic_Accuaracy,Logistic_Precision,Logistic_Recall]])

        # from sklearn.metrics import cohen_kappa_score
        kappa_Logistic = cohen_kappa_score(y_pred,y_test)

        arr = modelnew.best_estimator_.coef_
        newarr = arr.reshape(len(modelnew.best_estimator_.coef_[0]),)
        z = pd.Series(newarr,index=X.columns).sort_values(ascending=False)
        z = z.to_frame(name="Importance")
        z['Variable'] = z.index
        z['Variable'] = z['Variable'].str.split('_').str[0]
        z = z.reset_index(drop=True)
        z['Importance'] = z['Importance'].abs()

        z['Importance'] = z['Importance'] + abs(min(z['Importance'])) + 1
        z = z.sort_values('Importance',ascending=False)
        z = z.groupby(['Variable'], sort=False)['Importance'].max()
        z = z.to_frame(name="Importance")
        z['Variable'] = z.index
        z["x"] = (z['Importance'] / sum(z['Importance'])) * 100
        z['Variable'] = z.index
        z = z.reset_index(drop=True)
        z = z.rename(columns={'Variable': 'Group.1'}, inplace=False)

        z['s'] = 3
        z['Kappa'] = kappa_Logistic
        w1 = z.iloc[0:10]
        w1 = pd.DataFrame(w1[["s","Kappa","Group.1"]])
        w1
        zz = pd.DataFrame(z[["Group.1","x"]])

        # ranking based on the results
        tab0 = pd.concat([tab,tab_1,w1])
        # tab=tab0
        # data = [[Random_Forest_Accuaracy,Random_Forest_Precision,Random_Forest_Recall], [Boosting_Accuaracy,Boosting_Precision,Boosting_Recall],[Logistic_Accuaracy,Logistic_Precision,Logistic_Recall]]
        data = [[Random_Forest_Accuaracy,Random_Forest_Precision,Random_Forest_Recall], [Boosting_Accuaracy,Boosting_Precision,Boosting_Recall],
                [Logistic_Accuaracy,Logistic_Precision,Logistic_Recall]]
        columns = ["Accuracy","Precision","Recall"]
        dataframe_new_check_original = pd.DataFrame(data, columns=columns)

        dataframe_new_check_original['rank_Accuracy_original'] = dataframe_new_check_original['Accuracy'].rank(na_option='top')
        dataframe_new_check_original['rank_Precision_original'] = dataframe_new_check_original['Precision'].rank(na_option='top')
        dataframe_new_check_original['rank_Recall_original'] = dataframe_new_check_original['Recall'].rank(na_option='top')
        dataframe_new_final_original = pd.DataFrame(dataframe_new_check_original[["rank_Accuracy_original","rank_Precision_original","rank_Recall_original"]])
        dataframe_new_final_original = pd.DataFrame(dataframe_new_check_original[["rank_Accuracy_original","rank_Precision_original",
                                                                                    "rank_Recall_original"]])
        dataframe_new_final_original['mean'] = dataframe_new_final_original.mean(axis=1)

        dataframe_new_final_original['Overall_Rank'] = dataframe_new_final_original['mean'].rank(na_option='top')

        dataframe_new_final_original.rename(columns={'rank_Accuracy_original': 'rankAccuracy', 'rank_Precision_original': 'rankPrecision',
                                                        'rank_Recall_original': 'rankRecall'})

        # Obtaining the strengths of the results with completely new dataset#######
        # with Random forest

        # test_Set_Considered = n[~n.index.isin(training_sample.index)]
        test_Set_Considered_true = test_Set_Considered['Deleted']
        test_Set_Considered = test_Set_Considered.drop('Deleted', axis=1)
        pred4_newdataset = rfc.predict(test_Set_Considered)

        # from sklearn.metrics import precision_recall_fscore_support as score
        precision,recall,fscore,support = score(test_Set_Considered_true, pred4_newdataset,average='macro')

        # from sklearn.metrics import accuracy_score
        precision,recall,fscore,support = score(test_Set_Considered_true, pred4_newdataset,average='macro')

        rf_Accuracy = accuracy_score(test_Set_Considered_true, pred4_newdataset)

        confusion_matrix(test_Set_Considered_true, pred4_newdataset)

        Random_Forest_Accuaracy_new = rf_Accuracy
        Random_Forest_Precision_new = precision
        Random_Forest_Recall_new = recall

        Random_Forest_Summary_newdataset = pd.DataFrame(columns=["Random_Forest_Accuaracy","Random_Forest_Precision","Random_Forest_Recall"],
                                                        data=[[Random_Forest_Accuaracy_new,Random_Forest_Precision_new,Random_Forest_Recall_new]])

        # Boosting
        pred3_newdataset = clf.predict(test_Set_Considered)

        precision,recall,fscore,support = score(test_Set_Considered_true, pred3_newdataset,average='macro')

        clf_Accuracy = accuracy_score(test_Set_Considered_true, pred3_newdataset)

        confusion_matrix(test_Set_Considered_true, pred3_newdataset)

        Boosting_Accuaracy_new = clf_Accuracy
        Boosting_Precision_new = precision
        Boosting_Recall_new = recall

        Boosting_Summary_newdataset = pd.DataFrame(columns=["Boosting_Accuaracy","Boosting_Precision","Boosting_Recall"],
                                                    data=[[Boosting_Accuaracy_new,Boosting_Precision_new,Boosting_Recall_new]])


        # logistic regression
        probabilities_new = grid_search_cv.predict(test_Set_Considered)

        precision,recall,fscore,support = score(test_Set_Considered_true, probabilities_new,average='macro')

        log_Accuracy = accuracy_score(test_Set_Considered_true, probabilities_new)

        a = confusion_matrix(test_Set_Considered_true, probabilities_new)

        Logistic_Accuaracy_new = log_Accuracy
        Logistic_Precision_new = precision
        Logistic_Recall_new = recall

        Logistic_Summary_newdataset = pd.DataFrame(columns=["Logistic_Accuaracy",
                                                            "Logistic_Precision","Logistic_Recall"],
                                                    data=[[Logistic_Accuaracy_new,Logistic_Precision_new,Logistic_Recall_new]])

        # Ranking based on new dataset
        data = [[Random_Forest_Accuaracy_new,Random_Forest_Precision_new,Random_Forest_Recall_new],
                [Boosting_Accuaracy_new,Boosting_Precision_new,Boosting_Recall_new],
                [Logistic_Accuaracy_new,Logistic_Precision_new,Logistic_Recall_new]]
        columns = ["Accuracy","Precision","Recall"]
        dataframe_new_check = pd.DataFrame(data, columns=columns)

        dataframe_new_check['rank_Accuracy'] = dataframe_new_check['Accuracy'].rank(na_option='top')
        dataframe_new_check['rank_Precision'] = dataframe_new_check['Precision'].rank(na_option='top')
        dataframe_new_check['rank_Recall'] = dataframe_new_check['Recall'].rank(na_option='top')
        # dataframe_new_final = pd.DataFrame(dataframe_new_check[["rank_Accuracy","rank_Precision","rank_Recall"]])
        dataframe_new_final = pd.DataFrame(dataframe_new_check[["rank_Accuracy","rank_Precision",
                                                                "rank_Recall"]])
        dataframe_new_final['mean'] = dataframe_new_final.mean(axis=1)

        dataframe_new_final['Overall_Rank'] = dataframe_new_final['mean'].rank(na_option='top')

        dataframe_new_final.rename(columns={'rank_Accuracy': 'rankAccuracy', 'rank_Precision': 'rankPrecision',
                                            'rank_Recall': 'rankRecall'})

        tab['s'] = dataframe_new_final.iloc[0][4]
        tab['Accuracy'] = Random_Forest_Accuaracy
        tab_1['s'] = dataframe_new_final.iloc[1][4]
        tab_1['Accuracy'] = Boosting_Accuaracy
        w1['s'] = dataframe_new_final.iloc[2][4]
        w1['Accuracy'] = Logistic_Accuaracy

        tab = pd.concat([tab, tab_1,w1])
        tab0 = tab.sort_values("s", ascending=False)
        imp = tab0.drop(tab0.columns[[0, 1, 3]], axis=1)

        model_used_ = []
        model_accuracy_ = []

        # to select the model
        if (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[1][4]) & (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[2][4]):
            findata = s11
            model_used_ = rfc
            model_accuracy_ = Random_Forest_Accuaracy
        elif (dataframe_new_final.iloc[1][4] > dataframe_new_final.iloc[0][4]) & (dataframe_new_final.iloc[1][4] > dataframe_new_final.iloc[2][4]):
            findata = s33
            model_used_ = clf
            model_accuracy_ = Boosting_Accuaracy
        elif (dataframe_new_final.iloc[2][4] > dataframe_new_final.iloc[0][4]) & (dataframe_new_final.iloc[2][4] > dataframe_new_final.iloc[1][4]):
            findata = zz
            model_used_ = modelnew
            model_accuracy_ = Logistic_Accuaracy

        if (dataframe_new_final.iloc[0][4] == dataframe_new_final.iloc[1][4]) & (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[2][4]):
            if(dataframe_new_final.iloc[0][0] > dataframe_new_final.iloc[1][0]):
                findata = s11
                model_used_ = rfc
                model_accuracy_ = Random_Forest_Accuaracy
            elif(dataframe_new_final.iloc[0][0] < dataframe_new_final.iloc[1][0]):
                findata = s33
                model_used_ = clf
                model_accuracy_ = Boosting_Accuaracy
            elif(dataframe_new_final.iloc[0][0] == dataframe_new_final.iloc[1][0]):
                if(dataframe_new_final.iloc[0][1] > dataframe_new_final.iloc[1][1]):
                    findata = s11
                    model_used_ = rfc
                    model_accuracy_ = Random_Forest_Accuaracy
                elif(dataframe_new_final.iloc[0][1] < dataframe_new_final.iloc[1][1]):
                    findata = s33
                    model_used_ = clf
                    model_accuracy_ = Boosting_Accuaracy
                elif(dataframe_new_final.iloc[0][1] == dataframe_new_final.iloc[1][1]):
                    if(dataframe_new_final.iloc[0][2] > dataframe_new_final.iloc[1][2]):
                        findata = s11
                        model_used_ = rfc
                        model_accuracy_ = Random_Forest_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] < dataframe_new_final.iloc[1][2]):
                        findata = s33
                        model_used_ = clf
                        model_accuracy_ = Boosting_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] == dataframe_new_final.iloc[1][2]):
                        if(dataframe_new_final_original.iloc[0][4] > dataframe_new_final_original.iloc[1][4]):
                            findata = s11
                            model_used_ = rfc
                            model_accuracy_ = Random_Forest_Accuaracy
                        elif(dataframe_new_final_original.iloc[0][4] < dataframe_new_final_original.iloc[1][4]):
                            findata = s33
                            model_used_ = clf
                            model_accuracy_ = Boosting_Accuaracy

        if (dataframe_new_final.iloc[0][4] == dataframe_new_final.iloc[2][4]) & (dataframe_new_final.iloc[0][4] > dataframe_new_final.iloc[1][4]):
            if(dataframe_new_final.iloc[0][0] > dataframe_new_final.iloc[2][0]):
                findata = s11
                model_used_ = rfc
                model_accuracy_ = Random_Forest_Accuaracy
            elif(dataframe_new_final.iloc[0][0] < dataframe_new_final.iloc[2][0]):
                findata = zz
                model_used_ = modelnew
                model_accuracy_ = Logistic_Accuaracy
            elif(dataframe_new_final.iloc[0][0] == dataframe_new_final.iloc[2][0]):
                if(dataframe_new_final.iloc[0][1] > dataframe_new_final.iloc[2][1]):
                    findata = s11
                    model_used_ = rfc
                    model_accuracy_ = Random_Forest_Accuaracy
                elif(dataframe_new_final.iloc[0][1] < dataframe_new_final.iloc[2][1]):
                    findata = zz
                    model_used_ = modelnew
                    model_accuracy_ = Logistic_Accuaracy
                elif(dataframe_new_final.iloc[0][1] == dataframe_new_final.iloc[2][1]):
                    if(dataframe_new_final.iloc[0][2] > dataframe_new_final.iloc[2][2]):
                        findata = s11
                        model_used_ = rfc
                        model_accuracy_ = Random_Forest_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] < dataframe_new_final.iloc[2][2]):
                        findata = zz
                        model_used_ = modelnew
                        model_accuracy_ = Logistic_Accuaracy
                    elif(dataframe_new_final.iloc[0][2] == dataframe_new_final.iloc[2][2]):
                        if(dataframe_new_final_original.iloc[0][4] > dataframe_new_final_original.iloc[2][4]):
                            findata = s11
                            model_used_ = rfc
                            model_accuracy_ = Random_Forest_Accuaracy
                        elif(dataframe_new_final_original.iloc[0][4] < dataframe_new_final_original.iloc[2][4]):
                            findata = zz
                            model_used_ = modelnew
                            model_accuracy_ = Logistic_Accuaracy

        if (dataframe_new_final.iloc[1][4] == dataframe_new_final.iloc[2][4]) & (dataframe_new_final.iloc[1][4] > dataframe_new_final.iloc[0][4]):
            if(dataframe_new_final.iloc[1][0] > dataframe_new_final.iloc[2][0]):
                findata = s33
                model_used_ = clf
                model_accuracy_ = Boosting_Accuaracy
            elif(dataframe_new_final.iloc[1][0] < dataframe_new_final.iloc[2][0]):
                findata = zz
                model_used_ = modelnew
                model_accuracy_ = Logistic_Accuaracy
            elif(dataframe_new_final.iloc[1][0] == dataframe_new_final.iloc[2][0]):
                if(dataframe_new_final.iloc[1][1] > dataframe_new_final.iloc[2][1]):
                    findata = s33
                    model_used_ = clf
                    model_accuracy_ = Boosting_Accuaracy
                elif(dataframe_new_final.iloc[1][1] < dataframe_new_final.iloc[2][1]):
                    findata = zz
                    model_used_ = modelnew
                    model_accuracy_ = Logistic_Accuaracy
                elif(dataframe_new_final.iloc[1][1] == dataframe_new_final.iloc[2][1]):
                    if(dataframe_new_final.iloc[1][2] > dataframe_new_final.iloc[2][2]):
                        findata = s33
                        model_used_ = clf
                        model_accuracy_ = Boosting_Accuaracy
                    elif(dataframe_new_final.iloc[1][2] < dataframe_new_final.iloc[2][2]):
                        findata = zz
                        indata = zz
                        model_used_ = modelnew
                        model_accuracy_ = Logistic_Accuaracy
                    elif(dataframe_new_final.iloc[1][2] == dataframe_new_final.iloc[2][2]):
                        if(dataframe_new_final_original.iloc[1][4] > dataframe_new_final_original.iloc[2][4]):
                            findata = s33
                            model_used_ = clf
                            model_accuracy_ = Boosting_Accuaracy
                        elif(dataframe_new_final_original.iloc[1][4] < dataframe_new_final_original.iloc[2][4]):
                            findata = zz
                            model_used_ = modelnew
                            model_accuracy_ = Logistic_Accuaracy

        if(dataframe_new_final.iloc[0][4] == dataframe_new_final.iloc[1][4] == dataframe_new_final.iloc[2][4]):
            findata = s11
            model_used_ = rfc
            model_accuracy_ = Random_Forest_Accuaracy

        if model_used_ == []:
            model_used_ = rfc
            model_accuracy_ = Random_Forest_Accuaracy

        feature_list = findata['Group.1'].values
        data2 = pd.DataFrame()
        for u in range(len(feature_list)):
            data1 = pd.DataFrame(columns=['Corr'])
            # mylist =[]
            if (FinalData_re_path.columns.isin(feature_list).any()):
                corr = FinalData_re_path[feature_list[u]].corr(FinalData_re_path['Deleted'])
                a = [corr]
                df = pd.DataFrame(a,columns=["Correlation"])
                data2 = pd.concat([data2, df], ignore_index=True)

        findata["Correlation"] = data2["Correlation"]
        findata["Correlation"] = findata["Correlation"].fillna(0)

try:
    model_used_
except NameError:
    model_List
else:
    model_List.append(model_used_)

try:
    model_accuracy_
except NameError:
    Accuraicies
else:
    Accuraicies.append(model_accuracy_)

if(len(findata) >= 1):
    findata = findata.loc[findata['x'] > 0]
    findata['Portfolio'] = "Other"
else:
    findata = pd.DataFrame(columns=["Group.1","x","Portfolio","Correlation"])
    # findata = pd.DataFrame(columns=["Group.1","x","Portfolio"])

findata_new = pd.concat([findata_new, findata], ignore_index=True)
findata_new = findata_new.rename(columns={'Group.1': 'Variable','x': 'Influence'}, inplace=False)

if(len(findata_new) >= 1):
    findata_new.insert(0, 'RowNo', range(1, 1 + len(findata_new)))
    findata_new['CreatedDate'] = datetime.now()
    findata_new['UpdatedDate'] = datetime.now()
    findata_new = findata_new[['RowNo','Variable', "Influence",'Correlation','Portfolio','CreatedDate','UpdatedDate']]
else:
    findata_new = pd.DataFrame(columns=['RowNo','Variable', "Influence",'Correlation','Portfolio','CreatedDate','UpdatedDate'])

print(findata_new)
print(Accuraicies)
# Create a DataFrame for accuracy metrics
metric_table = pd.DataFrame(Accuraicies, columns=['Best Accuracy'])
if len(metric_table) == 0:
    metric_table['Best Accuracy'] = [0.0]
else:
    metric_table = metric_table.copy()

# Specify the local directory path
local_path_prefix = 'D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\model_data'  # Adjust as needed for your local environment
os.makedirs(local_path_prefix, exist_ok=True)

# Save metric table to CSV in the local Colab directory
metric_table_file = os.path.join(local_path_prefix, 'performance_metrics.csv')
metric_table.to_csv(metric_table_file)

# Print confirmation
print(f"Performance metrics saved to {metric_table_file}")

# Save the trained model locally
model_file = os.path.join(local_path_prefix, 'trained_model.pkl')
joblib.dump(value=model_List, filename=model_file)

# Print confirmation
print(f'Model trained and saved locally at {model_file}.')
