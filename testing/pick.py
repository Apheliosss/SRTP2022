import pandas as pd
from datetime import datetime
from tqdm import tqdm


# def getyear(str):
#     print(str)
#     date = datetime.strptime(str, "%Y/%m/%d")
#     return date.year


# def getmonth(str):
#     date = datetime.strptime(str, "%Y/%m/%d")
#     return date.month


def select(data, ratio):
    year = 0
    month = 0
    tdf = pd.DataFrame(columns=data.columns)
    positive = pd.DataFrame(columns=data.columns)
    nagative = pd.DataFrame(columns=data.columns)
    index = -1
    data.sort_values(by=['year', 'month'])
    for i in tqdm(range(0, 10000)):
        if not year == data.loc[i]['year'] or not month == data.loc[i]['month']:
            if not index == -1:
                tp = tdf.sample(frac=ratio)
                tn = tdf.append(tp)
                tn.drop_duplicates(keep=False, inplace=True)
                # print(tdf)
                # print(tp)
                # print(tn)
                positive = positive.append(tp)
                nagative = nagative.append(tn)
            tdf = pd.DataFrame(columns=data.columns)
            year = data.loc[i]['year']
            month = data.loc[i]['month']
            index = i
        tdf.loc[i-index] = data.loc[i]
    return positive, nagative


# data = pd.read_csv("../data.csv")
# tdf = data['date'].str.split('/', expand=True)
# tdf.columns = ['year', 'month', 'day']
# data['month'] = tdf['month']
# data['year'] = tdf['year']
# positive, nagative = select(data, 0.3)
# print(data.head())
# print(positive.head())
# print(nagative.head())
# positive.to_csv("../positive.csv",index=False)
# nagative.to_csv("../nagative.csv",index=False)
