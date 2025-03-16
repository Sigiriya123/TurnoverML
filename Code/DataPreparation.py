import pandas as pd
import numpy as np
from datetime import date

# Load data directly from CSV files without Azure dependencies
portfolios = pd.read_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\Portfolios.csv')
resda = pd.read_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\Resda.csv')
re2 = pd.read_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\Re2.csv')

# 2.3 Prepare the data
# data preparation
# data preparation for time series forecasting
resda["deleted"] = resda["deleted"].astype(int)
Treand_Seasonality_Prepared = pd.DataFrame(columns=["yrnmo","count","Name"])
Treand_Seasonality_Prepared.set_index('yrnmo',inplace=True)
groups_ts = resda.groupby(["Name"])
keys_ts = groups_ts.groups.keys()
for i in keys_ts:
        subsetData = groups_ts.get_group(i)
        subsetData = pd.DataFrame(subsetData)
        resda__ = pd.DataFrame(subsetData[['deleted','deleteddate','Name']])
        resda1 = pd.DataFrame(subsetData[['deleted','deleteddate']])
        start_yr = 2016
        date_range = pd.date_range(start='1/1/2016', end=date.today().replace(day=1), freq='MS')
        dates = date_range.to_pydatetime()
        first3_predictions = []
        for x in range(0,len(dates)):
            first3_predictions.append(dates[x].strftime('%Y-%m'))
        first3_predictions

        dataframe = pd.DataFrame(first3_predictions, columns=['yrnmo'])
        dataframe["count"] = 0
        
        # Ensure 'deleteddate' is in datetime format
        resda1['deleteddate'] = pd.to_datetime(resda1['deleteddate'], errors='coerce')

        # Now you can use .dt accessor safely
        resda1['deleteddate'] = resda1['deleteddate'].dt.to_period('M')

        resda2 = resda1[resda1['deleteddate'] >= dataframe['yrnmo'].iloc[0]]
        resda2 = resda2[resda2['deleted'] > 0]
        agr = resda2.groupby('deleteddate', as_index=False).agg({"deleted": "sum"})
        agr = agr.rename(columns={'deleted': 'count'}, inplace=False)
        agr = agr.rename(columns={'deleteddate': 'yrnmo'}, inplace=False)
        agr = pd.DataFrame(agr)
        agr['yrnmo'] = agr['yrnmo'].dt.strftime('%Y-%m')
        data_1a = dataframe.set_index('yrnmo')
        data_1a.update(agr.set_index('yrnmo'))
        data_out = data_1a.reset_index()
        data_out = data_out.fillna(0.00)
        data_out.yrnmo = pd.to_datetime(data_out.yrnmo)

        #data_out.set_index('yrnmo', inplace=True)
        data_out['Name'] = resda__['Name'].iloc[0]
        # data_out["Name"]=subsetData["Name"]
        Treand_Seasonality_Prepared = pd.concat([Treand_Seasonality_Prepared, data_out], ignore_index=True)  ## Original => ignore_index=True
        Treand_Seasonality_Prepared["count"] = Treand_Seasonality_Prepared["count"].astype(float)


# Data preparation for turnover probability calculation ############################### NO AZURE DEPENDENCIES FORM HERE ON

FinalData_re = pd.DataFrame()
FinalData_re1 = pd.DataFrame()
FinalData_re2 = pd.DataFrame()

groups_ts = re2.groupby(["Name"])
keys_ts = groups_ts.groups.keys()
for i in keys_ts:
    re2 = groups_ts.get_group(i)
    re2 = pd.DataFrame(re2)
    re2new = re2.iloc[:, :-1]
    # new addition
    re3 = re2new.copy()
    re3.insert(0, 'A', 0)
    re3.insert(0, 'B', 0)
    re1 = re2new.drop('number', axis=1)
    # re = re1.drop('EmployeeCode', axis=1)
    re = re1.drop('EmployeeCode', axis=1)
    if (len(re1.index) >= 1):
        # organizing date columns
        re_subset = pd.DataFrame(re[["DateOfMarriage","DateOfBirth","DateOfJoin","DateOfRetirement","LastWorkingDate","DeletedDate"]])
        now = date.today()
        re_subset = re_subset.fillna(now)
        re_subset['DateOfMarriage'] = pd.to_datetime(re_subset['DateOfMarriage']).dt.date
        re_subset['DateOfBirth'] = pd.to_datetime(re_subset['DateOfBirth']).dt.date
        re_subset['DateOfJoin'] = pd.to_datetime(re_subset['DateOfJoin']).dt.date
        re_subset['DateOfRetirement'] = pd.to_datetime(re_subset['DateOfRetirement']).dt.date
        re_subset['LastWorkingDate'] = pd.to_datetime(re_subset['LastWorkingDate']).dt.date
        re_subset['DeletedDate'] = pd.to_datetime(re_subset['DeletedDate']).dt.date

        re['DateOfMarriage'] = re_subset['DateOfMarriage']
        re['DateOfBirth'] = re_subset['DateOfBirth']
        re['DateOfJoin'] = re_subset['DateOfJoin']
        re['DateOfRetirement'] = re_subset['DateOfRetirement']
        re['LastWorkingDate'] = re_subset['LastWorkingDate']
        re['DeletedDate'] = re_subset['DeletedDate']

        re1['DateOfMarriage'] = re_subset['DateOfMarriage']
        re1['DateOfBirth'] = re_subset['DateOfBirth']
        re1['DateOfJoin'] = re_subset['DateOfJoin']
        re1['DateOfRetirement'] = re_subset['DateOfRetirement']
        re1['LastWorkingDate'] = re_subset['LastWorkingDate']
        re1['DeletedDate'] = re_subset['DeletedDate']

        # calculating AGE and removing columns
        re['Age'] = date.today().year - (pd.DatetimeIndex(re['DateOfBirth']).year)
        re.loc[(re['Age'] <= 0) | (re['Age'] >= 85) | (re['Age'].isna()), 'Age'] = 0
        re = re.drop(['DateOfBirth'], axis=1)
        re1['Age'] = date.today().year - (pd.DatetimeIndex(re1['DateOfBirth']).year)
        re1.loc[(re1['Age'] <= 0) | (re1['Age'] >= 85) | (re1['Age'].isna()), 'Age'] = 0
        re1 = re1.drop(['DateOfBirth'], axis=1)

        # Years after Marriage
        re['married'] = date.today().year - (pd.DatetimeIndex(re['DateOfMarriage']).year)
        re.loc[(re['married'] <= 0) | (re['married'] >= 85) | (re['married'].isna()), 'married'] = 0
        re = re.drop(['DateOfMarriage'], axis=1)
        re1['married'] = date.today().year - (pd.DatetimeIndex(re1['DateOfMarriage']).year)
        re1.loc[(re1['married'] <= 0) | (re1['married'] >= 85) | (re1['married'].isna()), 'married'] = 0
        re1 = re1.drop(['DateOfMarriage'], axis=1)

        # Years to Retire
        re['Retire'] = (pd.DatetimeIndex(re['DateOfRetirement']).year) - date.today().year
        re.loc[(re['Retire'] <= 0) | (re['Retire'].isna()), 'Retire'] = 0
        re = re.drop(['DateOfRetirement'], axis=1)
        re1['Retire'] = (pd.DatetimeIndex(re1['DateOfRetirement']).year) - date.today().year
        re1.loc[(re1['Retire'] <= 0) | (re1['Retire'].isna()), 'Retire'] = 0
        re1 = re1.drop(['DateOfRetirement'], axis=1)

        # Service Years
        re['Service'] = (pd.DatetimeIndex(re['LastWorkingDate']).year) - (pd.DatetimeIndex(re['DateOfJoin']).year)
        re.loc[(re['Service'] <= 0) | (re['Service'].isna()), 'Service'] = 0
        re = re.drop(['DateOfJoin'], axis=1)
        re = re.drop(['LastWorkingDate'], axis=1)
        re1['Service'] = (pd.DatetimeIndex(re1['LastWorkingDate']).year) - (pd.DatetimeIndex(re1['DateOfJoin']).year)
        re1.loc[(re1['Service'] <= 0) | (re1['Service'].isna()), 'Service'] = 0
        re1 = re1.drop(['DateOfJoin'], axis=1)
        re1 = re1.drop(['LastWorkingDate'], axis=1)

        # replacing the mode/most frequent values

        re['TravelModeCode'] = re['TravelModeCode'].fillna(re['TravelModeCode'].mode().iloc[0])
        re['ReligionCode'] = re['ReligionCode'].fillna(re['ReligionCode'].mode().iloc[0])
        re['MaritalStatus'] = re['MaritalStatus'].fillna(re['MaritalStatus'].mode())
        re['LivingCode'] = re['LivingCode'].fillna(re['LivingCode'].mode().iloc[0])
        re['EmployedCompanyCode'] = re['EmployedCompanyCode'].fillna(re['EmployedCompanyCode'].mode().iloc[0])
        re['EmploymentTypeCode'] = re['EmploymentTypeCode'].fillna(re['EmploymentTypeCode'].mode().iloc[0])
        re['GenderCode'] = re['GenderCode'].fillna(re['GenderCode'].mode().iloc[0])
        re['TravelModeCode'] = re['TravelModeCode'].astype('category')
        re['ReligionCode'] = re['ReligionCode'].astype('category')
        re['MaritalStatus'] = re['MaritalStatus'].astype('category')
        re['LivingCode'] = re['LivingCode'].astype('category')
        re['EmployedCompanyCode'] = re['EmployedCompanyCode'].astype('category')
        re['EmploymentTypeCode'] = re['EmploymentTypeCode'].astype('category')
        re['GenderCode'] = re['GenderCode'].astype('category')

        # Grade,joined type and reporting person
        re.loc[(re['ReportingPersonCode'].isna()), 'ReportingPersonCode'] = 99
        re['ReportingPersonCode'] = re['ReportingPersonCode'].astype('category')
        re.loc[(re['JoinedTypeCode'].isna()), 'JoinedTypeCode'] = 99
        re['JoinedTypeCode'] = re['JoinedTypeCode'].astype('category')
        re.loc[(re['GradeCode'].isna()), 'GradeCode'] = 99
        re['GradeCode'] = re['GradeCode'].astype('category')

        re1.loc[(re1['ReportingPersonCode'].isna()), 'ReportingPersonCode'] = 99
        re1['ReportingPersonCode'] = re1['ReportingPersonCode'].astype('category')
        re1.loc[(re1['JoinedTypeCode'].isna()), 'JoinedTypeCode'] = 99
        re1['JoinedTypeCode'] = re1['JoinedTypeCode'].astype('category')
        re1.loc[(re1['GradeCode'].isna()), 'GradeCode'] = 99
        re1['GradeCode'] = re1['GradeCode'].astype('category')

        # spouse disabled,spouse alive and deleted
        re.SpouseDisabled.fillna(value=np.nan, inplace=True)
        re.SpouseAlive.fillna(value=np.nan, inplace=True)
        re['SpouseDisabled'] = re['SpouseDisabled'].fillna(0)
        re['SpouseAlive'] = re['SpouseAlive'].fillna(1)
        re['SpouseDisabled'] = re['SpouseDisabled'].astype(int)
        re['SpouseAlive'] = re['SpouseAlive'].astype(int)
        re['SpouseDisabled'] = re['SpouseDisabled'].astype('category')
        re['SpouseAlive'] = re['SpouseAlive'].astype('category')
        re.Deleted.fillna(value=np.nan, inplace=True)
        re['Deleted'] = re['Deleted'].fillna(0)
        re['Deleted'] = re['Deleted'].astype(int)
        re['Deleted'] = re['Deleted'].astype('category')

        re1.SpouseDisabled.fillna(value=np.nan, inplace=True)
        re1.SpouseAlive.fillna(value=np.nan, inplace=True)
        re1['SpouseDisabled'] = re1['SpouseDisabled'].fillna(0)
        re1['SpouseAlive'] = re1['SpouseAlive'].fillna(1)
        re1['SpouseDisabled'] = re1['SpouseDisabled'].astype(int)
        re1['SpouseAlive'] = re1['SpouseAlive'].astype(int)
        re1['SpouseDisabled'] = re1['SpouseDisabled'].astype('category')
        re1['SpouseAlive'] = re1['SpouseAlive'].astype('category')
        re1.Deleted.fillna(value=np.nan, inplace=True)
        re1['Deleted'] = re1['Deleted'].fillna(0)
        re1['Deleted'] = re1['Deleted'].astype(int)
        re1['Deleted'] = re1['Deleted'].astype('category')

        # numeric variables
        re.loc[(re['PastEmployment'].isna()), 'PastEmployment'] = 0
        re.loc[(re['married'].isna()), 'married'] = 0
        re.loc[(re['FinalScore'].isna()), 'FinalScore'] = 0
        re.loc[(re['MinimumOvertimePayment'].isna()), 'MinimumOvertimePayment'] = 0
        re.loc[(re['NoOfChildren'].isna()), 'NoOfChildren'] = 0
        re.loc[(re['NoOfDependants'].isna()), 'NoOfDependants'] = 0
        re.loc[(re['Age'].isna()), 'Age'] = 0
        re.loc[(re['Retire'].isna()), 'Retire'] = 0
        re.loc[(re['Service'].isna()), 'Service'] = 0

        re1.loc[(re1['PastEmployment'].isna()), 'PastEmployment'] = 0
        re1.loc[(re1['married'].isna()), 'married'] = 0
        re1.loc[(re1['FinalScore'].isna()), 'FinalScore'] = 0
        re1.loc[(re1['MinimumOvertimePayment'].isna()), 'MinimumOvertimePayment'] = 0
        re1.loc[(re1['NoOfChildren'].isna()), 'NoOfChildren'] = 0
        re1.loc[(re1['NoOfDependants'].isna()), 'NoOfDependants'] = 0
        re1.loc[(re1['Age'].isna()), 'Age'] = 0
        re1.loc[(re1['Retire'].isna()), 'Retire'] = 0
        re1.loc[(re1['Service'].isna()), 'Service'] = 0

        # Spouse Occupation
        re.SpouseOccupation.fillna(value=np.nan, inplace=True)
        re['SpouseOccupation'] = re['SpouseOccupation'].fillna(0)
        re.loc[(re['SpouseOccupation'] != 0), 'SpouseOccupation'] = 1
        re['SpouseOccupation'] = re['SpouseOccupation'].astype('category')

        re1.SpouseOccupation.fillna(value=np.nan, inplace=True)
        re1['SpouseOccupation'] = re1['SpouseOccupation'].fillna(0)
        re1.loc[(re1['SpouseOccupation'] != 0), 'SpouseOccupation'] = 1
        re1['SpouseOccupation'] = re1['SpouseOccupation'].astype('category')

        # Replace specified values across both DataFrames
        replacement_dict = {
            '-': '', '/': '.', '^$|^ $': np.NaN, ' ': '_', '&': 'and', 'wife': np.NaN, 'N.A': np.NaN, 'student': np.NaN
        }

        for df in [re, re1]:
            df.replace(replacement_dict, regex=True, inplace=True)
            df = pd.DataFrame(df)

        re2new = re.copy()
        re11 = re
        re = re.drop(['DeletedDate'], axis=1)

        re['Name'] = re2['Name'].iloc[0]
        re1['Name'] = re2['Name'].iloc[0]

        FinalData_re = re
        FinalData_re1 = re1
        FinalData_re2 = re3

# List of columns to be converted to category
category_columns = [
    'Deleted', 'TravelModeCode', 'ReligionCode', 'MaritalStatus', 'LivingCode',
    'EmployedCompanyCode', 'EmploymentTypeCode', 'GenderCode', 'ReportingPersonCode',
    'JoinedTypeCode', 'GradeCode', 'SpouseDisabled', 'SpouseAlive', 'SpouseOccupation'
]

# Convert all specified columns to 'category' type in one go
FinalData_re[category_columns] = FinalData_re[category_columns].astype('category')

# Saving the new CSV files in the same directory as the script
portfolios.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\interData\Portfolio.csv', index=False)
Treand_Seasonality_Prepared.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\TreandSeasonalityPrepared.csv', index=False)
FinalData_re.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\inputData\FinalDataRe.csv', index=False)
FinalData_re1.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\interData\FinalData_re1.csv', index=False)
FinalData_re2.to_csv('D:\MiHCM Work\Performance Review Projects 2024\Code_copy\interData\FinalData_re2.csv', index=False)
