import streamlit as st
from datetime import date
import yfinance as yt
from fbprophet import Prophet

from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2011-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.image("/Users/krutarthpatel/Desktop/cibc.png",width=79,use_column_width=79, clamp=False, channels="RGB", output_format="auto")

st.title("Stock Prediction")

stocks = ("AAPL", "GOOG","MSFT","GME","AMZN")
selected_stock = st.selectbox("Select Stock For Prediction",stocks)

n_years=st.slider("Years Of Prediction:", 1,10)

period = n_years * 365

@st.cache
def load_data(ticker):
    data = yt.download(ticker, START,TODAY)
    data.reset_index(inplace=True)
    return data
    
data_load_state =st.text("Load Data...")
data= load_data(selected_stock)
data_load_state.text("Loading Data...Whoop Whoop Done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig =go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],name='Opening Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],name='Closing Price'))
    fig.layout.update(title_text="Time Series Data with Range-Slider", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecating

df_train= data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m= Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast=m.predict(future)


st.subheader('Predicted Data')
st.write(forecast.tail())

st.write('Prediction Graph')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write('Prediction Components')
fig2=m.plot_components(forecast)
st.write(fig2)



