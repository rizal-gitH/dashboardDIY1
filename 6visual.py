import streamlit as st  # Streamlit for creating the interactive app
import pandas as pd  # Pandas for data manipulation
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
import seaborn as sns  # Seaborn for advanced statistical plots

# Setting the page title and header
st.set_page_config(page_title="Dashboard DIY: Data Visualization", layout="centered")
st.header('Data Visualization using Matplotlib and Seaborn in Streamlit')

# load the data
df = pd.read_csv('./restaurant.csv')
st.subheader("Here is the first few rows of the dataset for reference:")
st.dataframe(df.head())

## Questions
# 1. Find number of Male and Female distribution (pie and bar)
# 2. Find distribution of Male and Female spent (boxplot or kdeplot)
# 3. Find distribution of averge total_bill across each day by male and female
# 4. Find the relation between total_bill and tip on time (scatter plot)

### Question 1: Male and Female Distribution (Pie and Bar Charts)
st.subheader('1.1 Distribution of Male and Female (Pie and Bar Charts)')
with st.container(border=True): 
    value_counts = df['sex'].value_counts() # Calculate the counts for each gender
    col1, col2 = st.columns(2)
    with col1: 
        st.subheader('Pie Chart')    
        # draw pie chart
        fig,ax = plt.subplots()
        ax.pie(value_counts,autopct='%0.2f%%',labels=['Male','Female'])
        st.pyplot(fig)
        
    with col2:
        st.subheader('Bar Chart')
        # draw bar plot
        fig,ax = plt.subplots()
        ax.bar(value_counts.index,value_counts)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)
    
    # put this in expander
    with st.expander('Click here to display value counts'):
        st.dataframe(value_counts)
   
# streamlit widgets and charts   
st.subheader('1.2 Distribution based on Feature Selected (Pie and Bar Charts)')  
data_types = df.dtypes
cat_cols = tuple(data_types[data_types == 'object'].index)
with st.container(border=True): 
    feature = st.selectbox('Select the feature you want to display bar and pie chart',
                           cat_cols
                           )
    value_counts = df[feature].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Pie Chart')    
        # draw pie chart
        fig,ax = plt.subplots()
        ax.pie(value_counts,autopct='%0.2f%%',labels=value_counts.index)
        st.pyplot(fig)
        
    with col2:
        st.subheader('Bar Chart')
        # draw bar plot
        fig,ax = plt.subplots()
        ax.bar(value_counts.index,value_counts)
        st.pyplot(fig)
    
    # put this in expander
    with st.expander('Click here to display value counts'):
        st.dataframe(value_counts)
        
        
## 2. Find distribution of Male and Female spent
st.subheader('2. Distribution of Spending by Male and Female')
with st.container(border=True):
    chart_types = ('Box Plot', 'Violin Plot', 'KDE Plot', 'Histogram',)
    chart_selection = st.selectbox('Select the type of chart to display spending distribution:',chart_types)
    # Create the selected chart
    fig, ax = plt.subplots()
    if chart_selection == 'Box Plot':
        sns.boxplot(x='sex', y='total_bill', data=df, ax=ax)
    elif chart_selection == 'Violin Plot':
        sns.violinplot(x='sex', y='total_bill', data=df, ax=ax)
    elif chart_selection == 'KDE Plot':
        sns.kdeplot(x=df['total_bill'], hue=df['sex'], ax=ax, shade=True)
    else:  # Histogram
        sns.histplot(x='total_bill', hue='sex', data=df, ax=ax, kde=True)
    st.pyplot(fig)
    
## 3. Find distribution of averge total_bill across each day by male and female
# bar, area, line
st.subheader('3.1 Average Total Bill by Day and Gender')
with st.container(border=True):
    features_to_groupby = ['day','sex']
    feature = ['total_bill']
    select_cols = feature+features_to_groupby
    avg_total_bill = df[select_cols].groupby(features_to_groupby).mean()
    avg_total_bill = avg_total_bill.unstack()
    # visual
    fig, ax = plt.subplots()
    avg_total_bill.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black')
    ax.set_ylabel('Average Total Bill')
    ax.set_title('Average Total Bill by Day and Gender')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

    st.dataframe(avg_total_bill)


st.subheader('3.2 Average Total Bill by Features (multiselect)')
with st.container(border=True):
    # 1. include all categorical features (multiselect)
    # 2. bar, area, line (selectbox)
    # 3. stacked (radio)
    c1, c2 , c3 = st.columns(3)
    with c1:
        group_cols = st.multiselect('select the features',cat_cols,cat_cols[0])
        features_to_groupby = group_cols
        n_features = len(features_to_groupby)    
    with c2:
        chart_type = st.selectbox('Select Chart type',
                                  ('bar','area','line')) 
    with c3:
        stack_option = st.radio('Stacked',('Yes','No'),)
        if stack_option == 'Yes':
            stacked = True
        else:
            stacked = False            

    feature = ['total_bill']
    select_cols = feature+features_to_groupby
    avg_total_bill = df[select_cols].groupby(features_to_groupby).mean()
    if n_features >1:
        for i in range(n_features-1):
            avg_total_bill = avg_total_bill.unstack()
            
    avg_total_bill.fillna(0,inplace=True)
    
    # visual
    fig, ax = plt.subplots()
    avg_total_bill.plot(kind=chart_type,ax=ax,stacked=stacked)
    ax.legend(loc='center left',bbox_to_anchor=(1.0,0.5))
    ax.set_ylabel('Avg Total Bill')
    st.pyplot(fig)

    with st.expander('click here to display values'):
        st.dataframe(avg_total_bill)
        
# 4. Find the relation between total_bill and tip on time (scatter plot)
st.subheader('4. Relationship between Total Bill and Tip')
hue_type = st.selectbox('Select a feature for coloring the scatter plot:',cat_cols)
# Scatter plot showing the relationship between 'total_bill' and 'tip'
fig, ax = plt.subplots()
sns.scatterplot(x='total_bill',y='tip',hue=hue_type,ax=ax,data=df)
ax.set_title('Scatter Plot: Total Bill vs Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
st.pyplot(fig)