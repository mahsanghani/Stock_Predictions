import yfinance as yf, pandas as pd, shutil, os, time, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from get_all_tickers import get_tickers as gt
from ta import add_all_ta_features
from ta.utils import dropna

tickers = gt.get_tickers_filtered(mktcap=150000, mktcap_max=10000000)
print("The amount of stock chosen to observe: " + str(len(tickers)))
shutil.rmtree("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\")
os.mkdir("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\")

i = API = Stock_Failure = Stocks_Not_Imported = 0;

while(i<len(tickers)) and (API<2000):
    try:
        stock=tickers[i]
        temp=yf.Ticker(str(stock))
        Hist_data=temp.history(period="max")
        Hist_data.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\"+stock+".csv")
        time.sleep(2)
        i+=1;API+=1;
        Stock_Failure=0
        print("Importing stock data:" + str(i))
    except ValueError:
        print("Yahoo Finance Backend Error, Attempting to Fix")
        if Stock_Failure>5:
            i+=1; Stocks_Not_Imported+=1;
        API+=1;Stock_Failure+=1;
print("The amount of stocks successfully imported: " + str(i-Stocks_Not_Imported))

shutil.rmtree("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\")
os.mkdir("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\")
list_files=(glob.glob("<Your Path>\\Bayesian_Logistic_Regression\\Stocks\\*.csv"))
for interval in list_files:
    Stock_Name = ((os.path.basename(interval)).split(".csv")[0])
    data = pd.read_csv(interval)
    dropna(data)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    data = data.iloc[100:]
    close_prices = data['Close'].tolist()
    five_day_obs = []
    thirty_day_obs = []
    sixty_day_obs = []
    x=0;
    while x < (len(data)):
        if x < (len(data)-5):
            if ((close_prices[x+1] + close_prices[x+1] + close_prices[x+1] + close_prices[x+1] + close_prices[x+1])/5) > close_prices[x]:
                five_day_obs.append(1)
            else:
                five_day_obs.append(0)
        else:
            five_day_obs.append(0)
        x+=1;
    y=0;
    while y < (len(data)):
        if y < (len(data)-30):
            thirtydaycalc=0;y2=0;
            while y2 < 30:
                thirtydaycalc = thirtydaycalc + close_prices[y+y2];y2+=1;
            if (thirtydaycalc/30) > close_prices[y]:
                thirty_day_obs.append(1)
            else:
                thirty_day_obs.append(0)
        else:
            thirty_day_obs.append(0)
        y+=1;
    z=0;
    while z < (len(data)):
        if z < (len(data)-60):
            sixtydaycalc=0;z2=0;
            while z2 < 60:
                sixtydaycalc = sixtydaycalc + close_prices[z+z2];z2+=1;
            if (sixtydaycalc/60) > close_prices[z]:
                sixty_day_obs.append(1)
            else:
                sixty_day_obs.append(0)
        else:
            sixty_day_obs.append(0)
        z+=1;
    data['Five_Day_Observation_Outcome'] = five_day_obs
    data['Thirty_Day_Observation_Outcome'] = thirty_day_obs
    data['Sixty_Day_Observation_Outcome'] = sixty_day_obs
    data.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\" + Stock_Name + ".csv")
    print("Data for " + Stock_Name + " has been substantiated with technical features.")

Hold_Results = []
list_files2 = (glob.glob("<Your Path>\\Bayesian_Logistic_Regression\\Stocks_Sub\\*.csv"))
for interval2 in list_files2:
    Stock_Name = ((os.path.basename(interval2)).split(".csv")[0])
    data = pd.read_csv(interval2,index_col=0)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    dependents = [data["Five_Day_Observation_Outcome"].to_list(),
                  data["Thirty_Day_Observation_Outcome"].to_list(),
                  data["Sixty_Day_Observation_Outcome"].to_list()];
    data = data.drop(['Five_Day_Observation_Outcome',
                      'Thirty_Day_Observation_Outcome',
                      'Sixty_Day_Observation_Outcome',
                      'Date','Open','High','Low','Close']);
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    Hold_Results_Section= []
    p=0;
    for dependent in dependents:
        x_train, x_test, y_train, y_test = train_test_split(data, dependent, test_size=0.2, random_state=0)
        model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0)
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)
        conf = confusion_matrix(y_test, y_prediction)
        if p==0: Hold_Results.append([Stock_Name, "Five_Day_Observation_Outcome",
            model.score(x_train, y_train), model.score(x_test, y_test), conf[0,0], conf[0,1], conf[1,0], conf[1,1]])
        if p==1: Hold_Results.append([Stock_Name, "Thirty_Day_Observation_Outcome",
            model.score(x_train, y_train), model.score(x_test, y_test), conf[0,0], conf[0,1], conf[1,0], conf[1,1]])
        if p==2: Hold_Results.append([Stock_Name, "Sixty_Day_Observation_Outcome",
            model.score(x_train, y_train), model.score(x_test, y_test), conf[0,0], conf[0,1], conf[1,0], conf[1,1]])
        p+=1;
    print("Model complete for " + Stock_Name)
df = pd.DataFrame(Hold_Results, columns=['Stock','Observation Period', 'Model Accuracy on Training Data',
    'True Positives', 'False Positives', 'False Negatives', 'True Negatives']);
df.to_csv("<Your Path>\\Bayesian_Logistic_Regression\\Model_Outcome.csv", index=False)