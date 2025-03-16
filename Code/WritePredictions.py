import joblib
import pyodbc
# import json
import pandas as pd
from datetime import datetime

import joblib

model_path = '/content/model_data/trained_model.pkl'
model = joblib.load(model_path)

FinalData_re_path_1 = pd.read_csv('/content/FinalDataRe_Copy.csv')
FinalData_re = FinalData_re_path_1.copy()
FinalData_re1 = FinalData_re.copy()

performanceMetric = pd.read_csv('/content/model_data/performance_metrics.csv')

FinalData_re['Deleted'] = FinalData_re['Deleted'].astype('category')
FinalData_re['TravelModeCode'] = FinalData_re['TravelModeCode'].astype('category')
FinalData_re['ReligionCode'] = FinalData_re['ReligionCode'].astype('category')
FinalData_re['MaritalStatus'] = FinalData_re['MaritalStatus'].astype('category')
FinalData_re['LivingCode'] = FinalData_re['LivingCode'].astype('category')
FinalData_re['EmployedCompanyCode'] = FinalData_re['EmployedCompanyCode'].astype('category')
FinalData_re['EmploymentTypeCode'] = FinalData_re['EmploymentTypeCode'].astype('category')
FinalData_re['GenderCode'] = FinalData_re['GenderCode'].astype('category')
FinalData_re['ReportingPersonCode'] = FinalData_re['ReportingPersonCode'].astype('category')
FinalData_re['JoinedTypeCode'] = FinalData_re['JoinedTypeCode'].astype('category')
FinalData_re['GradeCode'] = FinalData_re['GradeCode'].astype('category')
FinalData_re['SpouseDisabled'] = FinalData_re['SpouseDisabled'].astype('category')
FinalData_re['SpouseAlive'] = FinalData_re['SpouseAlive'].astype('category')
FinalData_re['SpouseOccupation'] = FinalData_re['SpouseOccupation'].astype('category')

Predict_Prep = pd.DataFrame()
Other_New1 = pd.DataFrame()
Other_New = pd.DataFrame()
Other_New_ = pd.DataFrame()

#FinalData_re['EmployeeCode'] = FinalData_re1['EmployeeCode']

# preparation of the dataframe
groups_ts = FinalData_re.groupby(["Name"])
keys_ts = groups_ts.groups.keys()
for i in keys_ts:
    subsetData = groups_ts.get_group(i)
    subsetData_ = pd.DataFrame(subsetData)
    subsetData = pd.DataFrame(subsetData)
    if(len(subsetData[subsetData['Deleted'] == 1]) >= 20) and (len(subsetData[subsetData['Deleted'] == 0]) >= 20):
        subsetData['Deleted'] = subsetData['Deleted'].astype('category')
        # subsetData["EmployeeCode"]=re2["EmployeeCode"]
        # subsetData=subsetData.drop(columns=["Name"])
        # n=pd.get_dummies(subsetData, columns=["EmployedCompanyCode","EmploymentTypeCode","GenderCode","GradeCode","LivingCode","MaritalStatusCode","JoinedTypeCode","ReligionCode","ReportingPersonCode","TravelModeCode","SpouseAlive","SpouseDisabled","SpouseOccupation"])
        Predict_Prep = pd.concat([Predict_Prep, subsetData], ignore_index=True)
    else:
        Other_New1 = pd.concat([Other_New1, subsetData], ignore_index=True)

if not Other_New1.empty and ('Deleted' in Other_New1.columns):
    if (len(Other_New1[Other_New1['Deleted'] == 1]) >= 20) and (len(Other_New1[Other_New1['Deleted'] == 0]) >= 20):
        Other_New1['Deleted'] = Other_New1['Deleted'].astype('category')
        Other_New1["Name"] = "Other"
        Final_Prepared_Prediction = Predict_Prep.append(Other_New1)
    else:
        Final_Prepared_Prediction = Predict_Prep
else:
    Final_Prepared_Prediction = Predict_Prep

    # calculation of turnover probabilities
if (len(model) == 0):
    Final_Prepared_Prediction = pd.DataFrame(columns=['RowNo','EmployeeCode','Score','Portfolio','Accuracy','CreatedDate','UpdatedDate'])
    rowmax = Final_Prepared_Prediction

else:
    groups_ts = Final_Prepared_Prediction.groupby(["Name"])
    keys_ts = groups_ts.groups.keys()
    y = list(keys_ts)
    # predicted_Data = pd.DataFrame(columns=["Date","Prediction"])
    for i in range(0,len(model)):
        Final_Prepared_Prediction_ = Final_Prepared_Prediction.loc[Final_Prepared_Prediction['Name'] == y[i]]
        Final_Prepared_Prediction_ = Final_Prepared_Prediction_.loc[Final_Prepared_Prediction_['Deleted'] == 0]
        # Keep a copy for later use
        Final_Prepared_Prediction_Copy = Final_Prepared_Prediction_.copy()

        # Drop unnecessary columns (if they exist)
        Final_Prepared_Prediction_ = Final_Prepared_Prediction_.drop(columns=['Deleted', 'Name', 'EmployeeCode'], errors='ignore')

        # Convert categorical variables
        n = pd.get_dummies(Final_Prepared_Prediction_,
                           columns=["EmployedCompanyCode", "EmploymentTypeCode", "GenderCode", "GradeCode",
                                    "LivingCode", "MaritalStatus", "JoinedTypeCode", "ReligionCode",
                                    "ReportingPersonCode", "TravelModeCode", "SpouseAlive",
                                    "SpouseDisabled", "SpouseOccupation"])

        # Model Prediction
        a = model[i].predict_proba(n)
        probs = a[:, 1]

        Final_Prepared_Prediction_Copy['Probability'] = probs
        Final_Prepared_Prediction_Copy['Accuracy'] = performanceMetric['Best Accuracy'].iloc[i]

        Other_New_ = pd.concat([Other_New_, Final_Prepared_Prediction_Copy], ignore_index=True)

    # Prepare final predicted DataFrame
    Predicted_final = pd.DataFrame(columns=["Name", "EmployeeCode", "Probability", "Accuracy"])
    #Other_New_ = Other_New_[["Name", "EmployeeCode", "Probability", "Accuracy"]]
    Other_New_ = Other_New_[["Name", "Probability", "Accuracy"]]
    Predicted_final = pd.concat([Predicted_final, Other_New_], ignore_index=True)

    # Rename columns correctly
    Predicted_final.rename(columns={'Name': 'Portfolio', 'Probability': 'Score'}, inplace=True)

    # Ensure 'Score' is float
    Predicted_final['Score'] = Predicted_final['Score'].astype('float64')

    # Add necessary columns
    Predicted_final.insert(0, 'RowNo', range(1, 1 + len(Predicted_final)))
    Predicted_final['CreatedDate'] = datetime.now()
    Predicted_final['UpdatedDate'] = datetime.now()

    # Final column selection
    Predicted_final = Predicted_final[['RowNo', 'EmployeeCode', 'Score', 'Portfolio', 'Accuracy', 'CreatedDate', 'UpdatedDate']]
    rowmax = Predicted_final

# Define CSV output path with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file_path = f'/content/MI_ALY_Turnover_Predictions_{timestamp}.csv'

# Save DataFrame to CSV
if not rowmax.empty:
    rowmax.to_csv(csv_file_path, index=False)
    print(f"CSV file saved successfully at {csv_file_path}")
else:
    print("No data to save. `rowmax` DataFrame is empty.")
    