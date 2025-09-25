
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

#@st.cache
def get_data():
  return pd.read_csv('data.csv')

def train_model():
  data = get_data()
  x = data.drop('MEDV', axis=1)
  y = data['MEDV']
  rf_regressor = RandomForestRegressor(random_state=42)
  rf_regressor.fit(x,y)
  return rf_regressor

#slide 51
data = get_data()
model = train_model()

#slide 52

st.title('AppIA - Prevendo preço de imóveis da cidade de Boston')

st.markdown('Este é um AppIA que prevê o preço de imóveis da cidade de Boston.')

st.subheader("Selecione os parâmetros do imóvel")

defaultcols = ['RM', 'PTRATIO', 'CRIM', 'MEDV']

cols = st.multiselect('Atributos', data.columns.tolist(), default=defaultcols)

#slide 53

st.dataframe(data[cols].head(10))

st.subheader('Distribuição de imóveis por preço')

faixa_valores = st.slider('Selecione a faixa de preço', float(data.MEDV.min()), 150., (10.0, 100.0))

dados = data[(data['MEDV'].between(left=faixa_valores[0], right= faixa_valores[1]))]

fig = px.histogram(dados, x='MEDV', nbins=100, title='Distribuição de imóveis por preço')
fig.update_xaxes(title='MEDV')
fig.update_yaxes(title='Total Imóveis')
st.plotly_chart(fig)

#slide 54

st.sidebar.subheader('Defina os atributos do imóvel para predição')

rm = st.sidebar.number_input('Número de quartos', value=1)
crim = st.sidebar.number_input('Taxa de criminalidade', value=data.CRIM.mean())
nox = st.sidebar.number_input('Concentração de óxido nítrico', value=data.NOX.mean())
ptratio = st.sidebar.number_input('Índice de alunos para professores', value=data.PTRATIO.mean())
indus = st.sidebar.number_input('Proporção de hectares de negócios', value=data.INDUS.mean())
chas = st.sidebar.selectbox('Faz limite com o rio?', ('Sim', 'Não'))

if chas == 'Sim':
  chas = 1
else:
  chas = 0

btn_predict = st.sidebar.button('Realizar Predição')

if btn_predict:
  result = model.predict([[rm, crim, nox, ptratio, indus, chas]])
  st.subheader('O valor previsto para o imóvel é:')
  result = 'US $ ' + str(round(result[0]*1000,2))
  st.write(result)
