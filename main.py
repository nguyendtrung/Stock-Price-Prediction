# Import libraries
from vnstock import *
from datetime import *
from dateutil.relativedelta import relativedelta as reladate
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

# Take the time
end_date = date.today()
start_date = end_date - reladate(years=3)
start_date = date(start_date.year, 1, 1)
end_date = end_date.strftime('%Y-%m-%d')
start_date = start_date.strftime('%Y-%m-%d')
print("start_date: {0}\nend_date: {1}".format(start_date, end_date))

# take hpg stock
scode = ['HPG'][0]
df_hpg = stock_historical_data(symbol=scode, start_date=start_date, end_date=end_date)
df_hpg.insert(0, column='scode', value=scode)
df_hpg = df_hpg.set_index('time')
df_hpg.index = pd.to_datetime(df_hpg.index)

# Split dataset
train_data, test_data = train_test_split(df_hpg, test_size=0.3, random_state=42, shuffle=False)


# Train model
model = ARIMA(train_data["close"], order=(1, 0, 2), seasonal_order=(0, 0, 0, 0))
arima = model.fit()

# Predict specific day
def predict_specific_day(day):
    predict = arima.predict(len(test_data) + len(train_data) + day)
    return predict.values

# Predict time series
def predict_time_series(day):
    start = len(test_data) + len(train_data)
    end = start + day
    predict = arima.predict(start=start, end=end)
    return predict.values


# main program
def main():
    st.title("Stock Price Prediction App")
    number = st.number_input("Enter Your Number", 0, 100, "min", 1)
    button1 = st.button("Predict a specific day")
    button2 = st.button("Predict time series")

    if button1:
        predict_value = predict_specific_day(number)
        st.write("The stock price as of {0} days from today".format(number))
        st.write(predict_value)
    elif button2:
        predict_value = predict_time_series(number)
        st.write("Stock price for next {0} days (include today):".format(number))
        st.write(predict_value)

if __name__ == '__main__':
    main()
