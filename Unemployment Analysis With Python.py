#!/usr/bin/env python
# coding: utf-8

# # UNEMPLOYMENT RATE PREDICTION IN INDIA USING PYTHON

# In[1]:


#Packages for plotting india map
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame


# In[2]:


#PACKAGES FOR DATA PROCESSING AND GRAPHS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import calendar
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#IMORTING DATA
data= pd.read_csv('Unemployment_Rate_upto_11_2020.csv',delimiter=',',skiprows=0,low_memory=False)
data.head()


# In[ ]:





# In[4]:


# Converting "Date" column to Datetime format
data[' Date']= pd.to_datetime(data[' Date'],dayfirst=True)

#Converting 'Frequency' and 'Region' columns to categorical data type
data[' Frequency'] = data[' Frequency'].astype('category')
data['Region'] = data['Region'].astype('category')


# In[5]:


data['Month']= data[' Date'].dt.month


# In[6]:


#converting 'month' to integer format
data['Month_int'] = data['Month'].apply(lambda x: int(x))

# Mapping integer month values to abbreviated month names
data['Month_name'] = data['Month_int'].apply(lambda x: calendar.month_abbr[x])
#Dropping the original 'Month' column
data.drop(columns='Month', inplace=True)
data['Month'] = data['Month_int'].apply(lambda x: calendar.month_abbr[x])


# In[7]:


# divided the Estimated employed by 100,000 for better aproximation 
data[' Estimated Employed']=data[' Estimated Employed']/100000


# # LOCATION OF INDIA ON WORLD MAP (IN RED)

# In[8]:


import geopandas as gpd
import matplotlib.pyplot as plt

# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the world map with India colored red
fig, ax = plt.subplots(figsize=(15, 10))
world.plot(ax=ax, color='lightgrey')  # Default color for all countries
world[world.name == "India"].plot(ax=ax, color='red')  # Color India red

# Set title and axis labels
plt.title('World Map with India Highlighted in Red')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Remove x and y axis for a cleaner look
ax.set_xticks([])
ax.set_yticks([])

plt.show()


# In[ ]:





# # Estimated Employed by Location

# In[9]:


# Correcting the column name and grouping the data by 'Location'
# Summing up the ' Estimated Employed' values
location_employment = data.groupby('Location')[' Estimated Employed'].sum().reset_index()

# Display the summed up employment figures for each location
location_employment


# In[ ]:





# In[10]:


# Grouping the data by 'location' instead and summing up the ' Estimated Employed' values
location1_employment = data.groupby('Location')[' Estimated Employed'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(location1_employment))

# Plotting the bar graph for location-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(location1_employment['Location'], location1_employment[' Estimated Employed'], color=[palette(i) for i in range(len(location1_employment))])

plt.title('Total Estimated Employment by Region (Descending Order)')
plt.xlabel('Location')
plt.ylabel('Estimated Employed')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Employed by Region

# In[ ]:





# In[11]:


# Correcting the column name and grouping the data by 'Location'
# Summing up the ' Estimated Employed' values
region1_employment = data.groupby('Region')[' Estimated Employed'].sum().reset_index()

# Display the summed up employment figures for each location
region1_employment.head()


# In[12]:


# Grouping the data by 'Region' and summing up the ' Estimated Employed' values
region1_employment = data.groupby('Region')[' Estimated Employed'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(region1_employment))

# Plotting the bar graph for region-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(region1_employment['Region'], region1_employment[' Estimated Employed'], color=[palette(i) for i in range(len(region1_employment))])

plt.title('Total Estimated Employment by Region (Descending Order)')
plt.xlabel('Region')
plt.ylabel('Estimated Employed')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Employed by Month

# In[13]:


# Correcting the column name and grouping the data by 'month'
# Summing up the ' Estimated Employed' values
month1_employment = data.groupby('Month')[' Estimated Employed'].sum().reset_index()

# Display the summed up employment figures for each month
month1_employment


# In[14]:


# Grouping the data by 'month' and summing up the ' Estimated Employed' values
month1_employment = data.groupby('Month')[' Estimated Employed'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(month1_employment))

# Plotting the bar graph for month-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(month1_employment['Month'], month1_employment[' Estimated Employed'], color=[palette(i) for i in range(len(month1_employment))])

plt.title('Total Estimated Employment by month (Descending Order)')
plt.xlabel('month')
plt.ylabel('Estimated Employed')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Unemployment Rate by Region

# In[15]:


# Correcting the column name and grouping the data by 'region'
# Summing up the ' Estimated unemployedment' values
region2_unemployment = data.groupby('Region')[' Estimated Unemployment Rate (%)'].sum().reset_index()

# Display the summed up unemployment figures for each region

region2_unemployment.head()


# In[16]:


# Grouping the data by 'Region' and summing up the ' Estimated unemployment rate' values
region2_employment = data.groupby('Region')[' Estimated Unemployment Rate (%)'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(region2_employment))

# Plotting the bar graph for region-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(region2_employment['Region'], region2_employment[' Estimated Unemployment Rate (%)'], color=[palette(i) for i in range(len(region2_employment))])

plt.title('Total Estimated Unemployment by Region (Descending Order)')
plt.xlabel('Region')
plt.ylabel('Estimated Unemployed rate')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Unemployment Rate by Location

# In[17]:


# Correcting the column name and grouping the data by 'Location'
# Summing up the ' Estimated Employed' values
loc_unemployment = data.groupby('Location')[' Estimated Unemployment Rate (%)'].sum().reset_index()

# Display the summed up employment figures for each location
loc_unemployment.head()


# In[18]:


# Grouping the data by 'Location' and summing up the ' Estimated unemployment' values
loc_employment = data.groupby('Location')[' Estimated Unemployment Rate (%)'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(loc_employment))

# Plotting the bar graph for location-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(loc_employment['Location'], loc_employment[' Estimated Unemployment Rate (%)'], color=[palette(i) for i in range(len(loc_employment))])

plt.title('Total Estimated Unemployment by Location (Descending Order)')
plt.xlabel('Location')
plt.ylabel('Estimated Unemployed rate')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Unemployment by month

# In[19]:


# Correcting the column name and grouping the data by 'month'
# Summing up the ' Estimated Employed' values
month2_unemployment = data.groupby('Month')[' Estimated Unemployment Rate (%)'].sum().reset_index()

# Display the summed up employment figures for each location
month2_unemployment


# In[20]:


# Grouping the data by 'month' and summing up the ' Estimated unemployment rate' values
month2_unemployment = data.groupby('Month')[' Estimated Unemployment Rate (%)'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(month2_unemployment))

# Plotting the bar graph for region-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(month2_unemployment['Month'], month2_unemployment[' Estimated Unemployment Rate (%)'], color=[palette(i) for i in range(len(month2_unemployment))])

plt.title('Total Estimated Unemployment by month (Descending Order)')
plt.xlabel('month')
plt.ylabel('Estimated Unemployed rate')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Labour Participation Rate by Region

# In[21]:


# Correcting the column name and grouping the data by 'region'
# Summing up the ' Estimated Employed' values
reg_part = data.groupby('Region')[' Estimated Labour Participation Rate (%)'].sum().reset_index()

# Display the summed up employment figures for each location
reg_part.head()


# In[22]:


# Grouping the data by 'Region' and summing up the Estimated Labour Participation Rate values
reg_unemployment = data.groupby('Region')[' Estimated Labour Participation Rate (%)'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(reg_unemployment))

# Plotting the bar graph for region-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(reg_unemployment['Region'], reg_unemployment[' Estimated Labour Participation Rate (%)'], color=[palette(i) for i in range(len(reg_unemployment))])

plt.title('Total Estimated Labour Participation Rate by Region (Descending Order)')
plt.xlabel('Region')
plt.ylabel('Estimated Labour Participation Rate')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Labour Participation Rate by Location

# In[23]:


# Correcting the column name and grouping the data by 'Location'
# Summing up the ' Estimated Employed' values
lo_part = data.groupby('Location')[' Estimated Labour Participation Rate (%)'].sum().reset_index()

# Display the summed up employment figures for each location
lo_part.head()


# In[24]:


# Grouping the data by 'Location' and summing up the Estimated Labour Participation Rate values
loc_employment = data.groupby('Location')[' Estimated Labour Participation Rate (%)'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(loc_employment))

# Plotting the bar graph for region-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(loc_employment['Location'], loc_employment[' Estimated Labour Participation Rate (%)'], color=[palette(i) for i in range(len(loc_employment))])

plt.title('Total Estimated Labour Participation Rate  by Location (Descending Order)')
plt.xlabel('Location')
plt.ylabel('Estimated Labour Participation Rate')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Estimated Labour Participation Rate by month

# In[25]:


# Summing up the ' Estimated Employed' values
m_part = data.groupby('Month')[' Estimated Labour Participation Rate (%)'].sum().reset_index()

# Display the summed up employment figures for each location
m_part.head()


# In[26]:


# Grouping the data by 'month' and summing up the Estimated Labour Participation Rate values
m_part = data.groupby('Month')[' Estimated Labour Participation Rate (%)'].sum().sort_values(ascending=False).reset_index()

# Creating a color palette with a distinct color for each bar
palette = plt.cm.get_cmap('plasma', len(m_part))

# Plotting the bar graph for region-wise estimated employment figures sorted in descending order
plt.figure(figsize=(15, 8))
bars = plt.bar(m_part['Month'], m_part[' Estimated Labour Participation Rate (%)'], color=[palette(i) for i in range(len(m_part))])

plt.title('Total Estimated Labour Participation Rate  by month (Descending Order)')
plt.xlabel('Month')
plt.ylabel('Estimated Labour Participation Rate')
plt.xticks(rotation=90)  # Rotate the x-axis labels to show clearly
plt.tight_layout()  # Adjust the layout to fit the labels
plt.show()


# In[ ]:





# # Comparing Estimated Labour Participation Rate and Estimated Unemployment Rate by region

# In[27]:


# Grouping the data by 'Region' and summing up both ' Estimated Labour Participation Rate (%)' and ' Estimated Unemployment Rate (%)' values
region_summary = data.groupby('Region').agg({
    ' Estimated Labour Participation Rate (%)': 'sum',
    ' Estimated Unemployment Rate (%)': 'sum'
}).sort_values(by=' Estimated Labour Participation Rate (%)', ascending=False).reset_index()

# Plotting the bar graph for region-wise sum of labour participation rate and unemployment rate
plt.figure(figsize=(15, 8))

# We need to plot two sets of bars, so we'll create an index for each set.
bar_width = 0.35  # width of the bars
index = np.arange(len(region_summary))

bar1 = plt.bar(index, region_summary[' Estimated Labour Participation Rate (%)'], bar_width,
               label='Labour Participation Rate (%)', color='royalblue')

bar2 = plt.bar(index + bar_width, region_summary[' Estimated Unemployment Rate (%)'], bar_width,
               label='Unemployment Rate (%)', color='tomato')

plt.xlabel('Region')
plt.ylabel('Rates (%)')
plt.title('Labour Participation Rate and Unemployment Rate by Region')
plt.xticks(index + bar_width / 2, region_summary['Region'], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# # comparing Estimated Labour Participation Rate  and Estimated Unemployment Rate  values by location

# In[28]:


# Grouping the data by 'Location' and summing up both ' Estimated Labour Participation Rate (%)' 
# and ' Estimated Unemployment Rate (%)' values
region_summary = data.groupby('Location').agg({
    ' Estimated Labour Participation Rate (%)': 'sum',
    ' Estimated Unemployment Rate (%)': 'sum'
}).sort_values(by=' Estimated Labour Participation Rate (%)', ascending=False).reset_index()

# Plotting the bar graph for location-wise sum of labour participation rate and unemployment rate
plt.figure(figsize=(15, 8))

# We need to plot two sets of bars, so we'll create an index for each set.
bar_width = 0.35  # width of the bars
index = np.arange(len(region_summary))

bar1 = plt.bar(index, region_summary[' Estimated Labour Participation Rate (%)'], bar_width,
               label='Labour Participation Rate (%)', color='royalblue')

bar2 = plt.bar(index + bar_width, region_summary[' Estimated Unemployment Rate (%)'], bar_width,
               label='Unemployment Rate (%)', color='tomato')

plt.xlabel('location')
plt.ylabel('Rates (%)')
plt.title('Labour Participation Rate and Unemployment Rate by location')
plt.xticks(index + bar_width / 2, region_summary['Location'], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# In[29]:


# Sunburst chart showing unemployment rate in each region and state

unemplo_df = data[['Location','Month', ' Estimated Unemployment Rate (%)', ' Estimated Employed', ' Estimated Labour Participation Rate (%)']]
unemplo = unemplo_df.groupby(['Location','Month'])[' Estimated Unemployment Rate (%)'].mean().reset_index()
fig = px.sunburst(unemplo, path=['Location','Month'], values=' Estimated Unemployment Rate (%)',
                  color_continuous_scale='Set1', title='Unemployment rate in each Region and Location',
                  height=650, template='ggplot2')
fig.show()


# # Estimated Unemployment Rate using latitude and Longitude representation on the india map  across location and within months

# In[30]:


#Impact of Lockdown on States Estimated Employed

fig = px.scatter_geo(data,'longitude', 'latitude', color="Location",
                     hover_name="Location", size=' Estimated Unemployment Rate (%)',
                     animation_frame="Month_name",scope='asia',template='seaborn',title='Estimated Unemployment Rate across Location')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.update_geos(lataxis_range=[10,30], lonaxis_range=[65, 100],oceancolor="lightblue",
    showocean=True)

fig.show()


# # Estimated Employed using latitude and Longitude representation on the india map within months

# In[31]:


fig = px.scatter_geo(data,'longitude', 'latitude', color="Location",
                     hover_name="Location", size=' Estimated Employed',
                     animation_frame="Month_name",scope='asia',template='seaborn',title='Estimated Employed across Location')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.update_geos(lataxis_range=[10,30], lonaxis_range=[65, 100],oceancolor="lightblue",
    showocean=True)

fig.show()


# In[ ]:





# # Estimated Labour Participation Rate using latitue and Longitude representation on the india map within months

# In[32]:


#

fig = px.scatter_geo(data,'longitude', 'latitude', color="Location",
                     hover_name="Location", size= ' Estimated Labour Participation Rate (%)',
                     animation_frame="Month_name",scope='asia',template='seaborn',title='Estimated Labour Participation Rate across Location')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.update_geos(lataxis_range=[10,30], lonaxis_range=[65, 100],oceancolor="lightblue",
    showocean=True)

fig.show()


# In[ ]:





# In[33]:


sns.pairplot(data, hue='Location', markers=["o", "s", "D"],vars=[ ' Estimated Unemployment Rate (%)',' Estimated Employed', ' Estimated Labour Participation Rate (%)'])
plt.show()


# In[ ]:





# In[ ]:





# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

# Setting up the matplotlib figure
plt.figure(figsize=(18, 10))

# Creating a list of the columns to plot
columns_to_plot = [
    'longitude', 
    'latitude', 
    ' Estimated Unemployment Rate (%)', 
    ' Estimated Employed', 
    ' Estimated Labour Participation Rate (%)'
]

# Creating a violin plot for each column
for i, column in enumerate(columns_to_plot):
    plt.subplot(2, 3, i+1)
    sns.violinplot(y=data[column], color=sns.color_palette("hls", 5)[i])
    plt.title(f'Violin Plot of {column}')


# Adjusting layout for better spacing
plt.tight_layout()
plt.show()


# In[ ]:





# # BUILDING THE MODEL

# In[35]:


#Libraries for building random forest classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[37]:


data.drop(columns=['Region', ' Date', ' Frequency','Month_name','Month'],axis=1,inplace=True)


# In[38]:


from sklearn.preprocessing import LabelEncoder
features = ['Location']
le = LabelEncoder()
for feature in features:
    le.fit(data[feature].unique())
    data[feature] = le.transform(data[feature])
    print(feature, data[feature].unique())


# ## CHECKING THE IMPORTANCE OF THE VARIOUS VARIABLES

# In[39]:


from sklearn.ensemble import RandomForestRegressor

# Separate the target variable (Salary) and the features
X = data.drop(' Estimated Unemployment Rate (%)', axis=1)
y = data[' Estimated Unemployment Rate (%)']

# Initialize and train a RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

importance_df


# In[51]:


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Define features and target variable
features = [' Estimated Employed', ' Estimated Labour Participation Rate (%)', 'Location', 'Month_int','longitude', 'latitude']
target = ' Estimated Unemployment Rate (%)'

# Prepare the data
X = data[features]
y = data[target]

# Train a Random Forest regressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X, y)

# Extract feature importances
feature_importances = rf.feature_importances_

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importances, align='center', color='GREEN')
plt.xlabel('Importance')
plt.title('Feature Importances using Random Forest')
plt.gca().invert_yaxis()  # Display the feature with the highest importance at the top
plt.show()


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(' Estimated Unemployment Rate (%)', axis=1), data[' Estimated Unemployment Rate (%)'], test_size=0.2, random_state=42)


# In[49]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the random forest regressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Fit the random forest model to the scaled training data
rf.fit(X_train, y_train)

# Predict on the scaled test set
y_pred_rf = rf.predict(X_test)

# Calculate performance metrics for the random forest model
MSE= mean_squared_error(y_test, y_pred_rf)
RMSE = mse_rf ** 0.5
R_Squared = r2_score(y_test, y_pred_rf)

MSE, RMSE, R_Squared


# Mean Squared Error (MSE): This is the average of the squares of the errors. The error is the difference between the actual values (from y_test) and the predicted values (from y_pred_rf

# Root Mean Squared Error (RMSE): This is the square root of the mean squared error. It's a measure of the average magnitude of the error, giving you an idea of how far the predictions tend to be from the actual values.

# R-squared (R²): This is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables in a regression model. It provides an indication of the goodness of fit of a set of predictions to the actual values. An R² of 1 indicates that the regression predictions perfectly fit the data

# In[56]:


import pandas as pd

# Create a DataFrame with actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})

# Optionally, reset the index if y_test is not already a simple range index
comparison_df.reset_index(drop=True, inplace=True)

# Print the DataFrame
print(comparison_df)


# # Difference between the Actual and predicted 

# In[55]:


# Calculate the difference between actual and predicted values
comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']

# Display the updated DataFrame
comparison_df


# In[57]:


import matplotlib.pyplot as plt

# Assuming you have already created comparison_df as shown previously
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
comparison_df.reset_index(drop=True, inplace=True)

# Plotting the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Actual'], label='Actual', color='blue', marker='o')
plt.plot(comparison_df['Predicted'], label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()


# # MODEL INTERPRETATION

#  The RMSE of 5.63 suggests that, on average, the model's predictions deviate from the actual values by about 5.63 percentage points.

#  An R² value of 0.66 means that approximately 66% of the variance in the unemployment rate is explained by the model. This is a relatively good score, indicating that the model has a decent fit to the data.

# In[ ]:




