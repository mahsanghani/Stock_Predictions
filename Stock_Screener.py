import smtplib
import ssl
from get_all_tickers import get_tickers as gt
import yfinance as yf, pandas as pd, shutil, os, time, glob

tickers = gt.get_tickers_filtered(mktcap=150000, mktcap_max=10000000)
print("The amount of stock chosen to observe: " + str(len(tickers)))
shutil.rmtree("<Your Path>\\Daily_Stock_Report\\Stocks\\")
os.mkdir("<Your Path>\\Daily_Stock_Report\\Stocks\\")

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
i=0;new_data=[];
list_files=(glob.glob("<Your Path>\\Daily_Stock_Report\\Stocks\\*.csv"))
while i < len(list_files):
    data = pd.read_csv(list_files[i]).tail(10)
    j=0;obv_value=0;pos_move=[];neg_move=[];
    while (j<10):
        if data.iloc[j,1] < data.iloc[j,4]:
            pos_move.append(j)
        elif data.iloc[j,1] > data.iloc[j,4]:
            neg_move.append(j)
        j += 1
    k = 0
    for l in pos_move: obv_value = round(obv_value + (data.iloc[l,5]/data.iloc[i,1]))
    for m in neg_move: obv_value = round(obv_value - (data.iloc[i,5]/data.iloc[i,1]))
    Stock_Name = ((os.path.basename(list_files[i])).split(".csv")[0])
    new_data.append([Stock_Name, obv_value])
    i+=1;
df = pd.DataFrame(new_data, columns=['Stock','OBV_Value'])
df["Stocks_Ranked"] = df["OBV_Value"].rank(ascending=False)
df.sort_values("OBV_Value", inplace=True, ascending=False)
df.to_csv("<Your Path>\\Daily_Stock_Report\\OBV_Ranked.csv", index=False)
Analysis = pd.read_csv("<Your Path>\\Daily_Stock_Report\\OBV_Ranked.csv")
top10 = Analysis.head(10)
bottom10 = Analysis.tail(10)
email="""\
Subject: Daily Stock Report
Your highest ranked OBV stocks of the day:
""" + top10.to_string(index=False) + """\
Your lowest ranking OBV stocks of the day:
""" + bottom10.to_string(index=False) + """\
Sincerely,
BOT
"""

context = ssl.create_default_context()
port=465
with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login("<Your email>", "<Your email password>")
    server.sendmail("<Your email>", "<Email receiving message>", email)