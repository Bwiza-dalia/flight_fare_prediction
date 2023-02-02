import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from utils import get_encoded_label

st.set_page_config(layout="wide")

# importing dataset
dataset = pd.read_csv('data_case2/final_unlabelled.csv')
finaldf = pd.read_csv('data_case2/final_labelled.csv')
model = joblib.load('./models_case2/flight_ticket_price_model.sav')

st.title("Flight Fare Prediction using Machine learning")


def description_function():
    st.subheader('Introduction')
    st.write(
        'In this project, we will be analyzing the flight fare prediction using Machine Learning dataset using'
        ' essential exploratory data analysis techniques then will draw some predictions about the price of the'
        ' flight based on some features such as what type of airline it is, what is the arrival time, what is the'
        ' departure time, what is the duration of the flight, source, destination and more.')
    st.write(
        'In commercial air transport, it deals with the fairness of tickets for journeys, which raised this project.')

    st.subheader('Project Statement')
    st.write(
        'Flight ticket prices can be something hard to guess. We have been provided with prices of flight tickets'
        ' for various airlines in particular period of time and between various cities, using which we aim to build '
        'a model which predicts the prices of the flights using various input features.')


# analysis function
def analysis_function(df):
    col1, col2 = st.columns(2)
    # displaying data in relation to airlines
    with col1:
        st.markdown('##### The relationship of airline and price')
        st.plotly_chart(px.box(data_frame=df, x='airline', y='price'))
    with col2:
        st.markdown('##### Price distribution')
        st.plotly_chart(ff.create_distplot(
            [df.price], group_labels=['price'], bin_size=20))

    cola, colb = st.columns(2)
    with cola:
        st.markdown('##### class,flight and price')
        st.pyplot(sns.catplot(data=df, x='airline', y='price',
                              hue='class', kind='bar', height=5, aspect=10 / 6))
    with colb:
        st.markdown('##### departure_day,flight and price')
        st.pyplot(sns.catplot(data=df, x='departure_dayname', y='price',
                              hue='airline', kind='bar', height=5, aspect=10 / 6))
    # Column 1

    colx, coly = st.columns(2)
    with colx:
        # st.set_column_headers("Ticket Type and Price Analysis")
        # st.write("This column shows the relationship between ticket_type and price.")
        st.markdown('##### Ticket Type vs Price')
        st.bar_chart(data=df, x='ticket_type', y='price')

    # Column 2
    with coly:
        st.markdown('##### Count of Flights Month Wise')
        dataset_new = pd.DataFrame(df['booking_month'].value_counts(sort=True, ascending=True))
        dataset_new.reset_index(inplace=True)

        dataset_new.rename(columns={'booking_month': 'flights count', 'index': 'Month'}, inplace=True)

        st.plotly_chart(px.bar(data_frame=dataset_new, x='Month', y='flights count'))
        # st.set_column_headers("Count of Flights Month Wise")
        # st.write("This column shows the count of flights month wise.")
        # st.markdown('##### Count of Flights Month Wise')
        # st.bar_chart(data=dataset, x='booking_month', y=dataset['airline'].value_counts())
        # # st.legend("Month Wise Count")
        # st.title("Count of Flights Month Wise")
        # st.caption("This chart shows the count of flights month wise.")
        # st.write("This chart shows that the count of flights is higher during the months of June and July.")
    colm, coln = st.columns(2)
    with colm:
        st.markdown('##### relationship between flights and price')
        st.plotly_chart(px.scatter(dataset, x='airline', y='price'))

    # displaying data
    st.markdown('#### Source data', unsafe_allow_html=True)
    st.dataframe(dataset.head())


# last page


def ai_function():
    st.header('User input for flights fare prediction')
    st.write('Please select input and press submit to predict the ticket fare')
    sel1, sel2 = st.columns(2)

    with st.form("ai form"):
        inputs = dict()
        labels = dict()
        with sel1:
            airline = st.selectbox('choose airline', (set(dataset['airline'])), key="1")
            source = st.selectbox('choose True Origin', (set(dataset['actual_source'])), key="2")
            destination = st.selectbox('choose True Destination', (set(dataset['actual_destination'])), key="3")
            class_cat = st.selectbox('choose Class Category', (set(dataset['class'])), key="4")

        with sel2:
            # stops = st.selectbox('choose number of stops', (set(dataset['estimated_stops'])), key="6")
            days_left = st.number_input("Enter days left to flight", min_value=1, max_value=100, step=1,
                                        key=12)
            # booking_day = st.selectbox('choose booking day of the week', (set(dataset['booking_day'])), key="7")
            # departure_day = st.selectbox('choose departure day of the year', (set(dataset['departure_day'])), key="8")
            ticket_type = st.selectbox('choose ticket type (return or one-way)', (set(dataset['ticket_type'])), key="9")
            departure_dayname = st.selectbox('choose departure day of week', (set(dataset['departure_dayname'])),
                                             key="10")

        # submission form
        submitted = st.form_submit_button("Submit")
        if submitted:
            inputs['airline'] = airline
            inputs['source'] = source
            inputs['destination'] = destination
            inputs['class'] = class_cat
            inputs['ticket_type'] = ticket_type
            inputs['departure_dayname'] = departure_dayname
            inputs['days_left'] = days_left

            # labelled values
            labels['airline'] = get_encoded_label(inputs['airline'], 'airline', dataset, finaldf)
            labels['source'] = get_encoded_label(inputs['source'], 'actual_source', dataset, finaldf)
            labels['destination'] = get_encoded_label(inputs['destination'], 'actual_destination', dataset, finaldf)
            labels['class'] = get_encoded_label(inputs['class'], 'class', dataset, finaldf)
            labels['departure_dayname'] = get_encoded_label(inputs['departure_dayname'], 'departure_dayname', dataset,
                                                            finaldf)
            labels['ticket_type'] = get_encoded_label(inputs['ticket_type'], 'ticket_type', dataset, finaldf)
            labels['days_left'] = inputs['days_left']
            # predicting for the model
            x_test = np.array(
                [labels['airline'], labels['source'], labels['departure_dayname'], labels['destination'],
                 labels['ticket_type'], labels['class'], labels['days_left']])
            x_test = x_test.reshape(1, -1)
            prediction = model.predict(x_test)

            st.subheader('Predicted Price in USD($)')
            st.subheader('$ ' + str(int(prediction[0])))


# creating sidebar menu
with st.sidebar:
    selected = option_menu(menu_title='Menu',
                           options=['Project Description', 'Analysis & Visualization', 'Predict your Fare'],
                           default_index=1)

if selected == 'Project Description':
    st.subheader('Project Description')
    description_function()

if selected == 'Analysis & Visualization':
    st.subheader('Data Exploration ')
    analysis_function(dataset)

if selected == 'Predict your Fare':
    ai_function()
