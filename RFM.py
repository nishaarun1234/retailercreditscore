# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:32:23 2022

@author: arunk
"""

## RFM
#### Credit Scoring Analysis Model for Customers/Retailers####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#lets import the data
creditanalysis = pd.read_excel("C:/Users/arunk/OneDrive/Desktop/clustering algo 72/CreditAnalysis_data.xlsx")
creditanalysis.info()
creditanalysis.head()

creditanalysis.columns



#lets remove the not required variables
creditanalysis = creditanalysis.drop(['Unnamed: 0', 'master_order_status', 'order_status'], axis=1)


## checking null values
creditanalysis.isna().sum()  # found  null values

##drop null values
creditanalysis.dropna(inplace=True)

## checking duplicates
duplicate = creditanalysis.duplicated()
duplicate
sum(duplicate)

# Removing Duplicates
creditanalysis = creditanalysis.drop_duplicates()




from sklearn.preprocessing import RobustScaler, LabelEncoder
#Label encoding for categorical features
cat_col = ["prod_names", "group", "dist_names", "retailer_names"]

lab = LabelEncoder()
mapping_dict ={}
for col in cat_col:
    creditanalysis[col] = lab.fit_transform(creditanalysis[col])
 
    le_name_mapping = dict(zip(lab.classes_,
                        lab.transform(lab.classes_))) #To find the mapping while encoding
 
    mapping_dict[col]= le_name_mapping
print(mapping_dict)


creditanalysis = creditanalysis.iloc[:, [0,1,2,3,4,5,6,7,8,10,9]]
creditanalysis.columns

# Number of records with negative quantity
creditanalysis.ordereditem_quantity[creditanalysis.ordereditem_quantity < 0 ].count() # no records

# Number of records with negative Unit Price 
creditanalysis.ordereditem_unit_price_net[creditanalysis.ordereditem_unit_price_net < 0].count() # no records

# Transforming created column to datetime type and mapping to a new columne as createddate
creditanalysis['created'] = pd.to_datetime(creditanalysis['created'])

# Construct Year, Month and YearMonth from Invoice Date field
creditanalysis['Year'], creditanalysis['Month'] = creditanalysis['created'].dt.year, creditanalysis['created'].dt.month
creditanalysis['YearMonth'] = creditanalysis['created'].map(lambda x: 100*x.year + x.month)

# Create "Date" column in datetime format to use for index
creditanalysis['Date'] = pd.to_datetime(creditanalysis.created.dt.date)
creditanalysis.set_index('Date', inplace=True)


# Counting types of Invoices
creditanalysis.master_order_id.value_counts()

# Unique values
creditanalysis.master_order_id.unique()

## Data Inspection and visualisations
# Review which countries do the orders come from
creditanalysis.retailer_names.value_counts()[0:10].plot(kind='barh');
plt.title('Number of orders per Country')
plt.xlabel('ordereditem_quantity');
plt.ylabel('Count');

#####################################
##Modelling - RFM Modelling

#In order to do Customer Segmentation, the RFM modelling technique will be used.

#RFM stands for Recency - Frequency - Monetary Value with the following definitions:

#Recency - Given a current or specific date in the past, when was the last time that the customer made a transaction
#Frequency - Given a specific time window, how many transactions did the customer do during that window
#Monetary Value or Revenue - Given a specific window, how much did the customer spend

## Recency Score

# Generate new dataframe based on unique CustomerID to keep track of RFM scores
customer = pd.DataFrame(creditanalysis['retailer_names'])
customer.columns = ['retailer_names']


# Generate new data frame based on latest Invoice date from retail_ppp dataframe per Customer (groupby = CustomerID)
recency = creditanalysis.groupby('retailer_names').created.max().reset_index()
recency.columns = ['retailer_names','LastPurchaseDate']

# Set observation point as the last invoice date in the dataset
LastInvoiceDate = recency['LastPurchaseDate'].max()

# Generate Recency in days by subtracting the Last Purchase date for each customer from the Last Invoice Date
recency['Recency'] = (LastInvoiceDate - recency['LastPurchaseDate']).dt.days

# Consolidate to retailer DataFrame
retailer = pd.merge(customer, recency[['retailer_names','Recency']], on='retailer_names')
retailer.head()

# Review statistics around Recency score 
retailer.describe()

# Plot Recency
retailer.Recency.plot.hist();
plt.xlabel("Recency in days")
plt.ylabel("Number of Customers")
plt.title("Recency Histogram");

creditanalysis.columns
##Frequency Score
#Frequency metric reflects the number of orders per retailer, so a simple count of the orders grouped per retailer_names would do

# Count number of invoices per retailer_names and store in new frequency Dataframe
frequency = creditanalysis.groupby('retailer_names').created.count().reset_index()
frequency.columns = ['retailer_names','Frequency']

# Consolidate Frequency to existing retailer DataFrame
retailer = pd.merge(retailer, frequency, on='retailer_names')

retailer.head()
retailer.Frequency.shape


# Plot Frequency
# Frequency seems to have some outliers, with vey high frequency, but very few in numbers
# In order to plot effectively and not have a skewed diagram, we've sorted the frequencies
# and cropped the top 72 values in our diagram
retailer.Frequency.sort_values().head(4300).plot.hist();
plt.xlabel("Frequency in days")
plt.ylabel("Number of Customers")
plt.title("Frequency Histogram");

##Monetery Value Score (Revenue)

##Revenue per transaction (new feature)
creditanalysis['Revenue'] = creditanalysis.ordereditem_unit_price_net * creditanalysis.ordereditem_quantity
creditanalysis.head()

# Revenue per transaction has already been calculated as per KPIs section
# Grouping revenue per Customer ID
revenue = creditanalysis.groupby('retailer_names').Revenue.sum().reset_index()

# Consolidate Revenue to existing Customer DataFrame
retailer = pd.merge(retailer, revenue, on='retailer_names')
retailer.head()

# Plot Revenue
retailer.Revenue.sort_values().head(4200).plot.hist();
plt.xlabel("Revenue in days")
plt.ylabel("Number of Customers")
plt.title("Revenue Histogram");


fig, (ax4, ax5, ax6) = plt.subplots(3)
fig.suptitle('Histograms')
retailer.Recency.plot.hist(ax = ax4, figsize = (12,12));
retailer.Frequency.sort_values().head(4300).plot.hist(ax = ax5);
retailer.Revenue.sort_values().head(4200).plot.hist(ax = ax6);
ax4.set_ylabel('Recency')
ax5.set_ylabel('Frequency')
ax6.set_ylabel('Revenue')


# Finally lets review the scatter plots between the different scores

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Scatter plots between RFM scores')
retailer.plot.scatter(x = 'Recency', y = 'Frequency', ax = ax1, figsize = (12,10));
retailer.plot.scatter(x = 'Recency', y = 'Revenue', ax = ax2);
retailer.plot.scatter(x = 'Frequency', y = 'Revenue', ax = ax3);

## K-Means Clustering

# Creating input features variable
X = retailer.loc[:, 'Recency':'Revenue']
X.head()

## Number of clusters (Configurable) - Initially we're attempting based on the 3 Segments (Low, Mid, High)
k=5

# Scaling input using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xstd = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Run and train main K-means algorithm based on all input features
from sklearn.cluster import KMeans
model = KMeans(n_clusters=k, random_state=0)
model.fit(Xstd)

# Review output cluster labels
cluster_labels = model.labels_
print("Assigned cluster labels: \n", cluster_labels)


# Review Centroids of clusters
centroids = model.cluster_centers_
print("Location of centroids: ")
print(centroids)

centroids[:,0]

# Append clusters to input features table
Xstd['clusters'] = cluster_labels
Xstd.head()

# Scatter plot of data coloured by cluster they belong to
fig, (ax4, ax5, ax6) = plt.subplots(3);
fig.suptitle('Scatter Plot of Segments based on RFM scores');

Xstd.plot.scatter(x = 'Recency', y = 'Frequency', c=Xstd['clusters'], colormap='viridis', ax=ax4, colorbar=False,figsize = (12,10));
ax4.scatter(centroids[:,0], centroids[:,1], marker='o', s=350, alpha=.8, c=range(0,k), 
            cmap='viridis');

Xstd.plot.scatter(x = 'Recency', y = 'Revenue', c=Xstd['clusters'], colormap='viridis', ax=ax5, colorbar=False);
ax5.scatter(centroids[:,0], centroids[:,2], marker='o', s=350, alpha=.8, c=range(0,k), 
            cmap='viridis');

Xstd.plot.scatter(x = 'Frequency', y = 'Revenue', c=Xstd['clusters'], colormap='viridis', ax=ax6, colorbar=False);
ax6.scatter(centroids[:,1], centroids[:,2], marker='o', s=350, alpha=.8, c=range(0,k), 
            cmap='viridis');

## Evaluation Metrics - Silhouette and Inertia scores

from sklearn import metrics
metrics.silhouette_score(Xstd, cluster_labels, metric='euclidean') #0.7310180340306249

model.inertia_ # 5072.270363731566

##Retailer Dataframe and Visualization

retailer['Cluster'] = cluster_labels
retailer.Cluster.unique()


retailer['Profile'] = cluster_labels
retailer['Profile'].replace({0: "poor", 1: "fair", 2: "good", 3:"verygood", 4:"excellent"}, inplace = True)

retailer.Profile.describe(include=['O'])


retailer.groupby('Profile').Profile.count().plot.bar()
plt.xlabel("Customer Segments")
plt.ylabel("Number of Customers")
plt.title("Segments Summary");

# appending multiple DataFrame

creditanalysis = creditanalysis.reset_index()
finaldata3 = pd.concat([creditanalysis, retailer], axis=1)

finaldata3.columns

## apply algorithm on new dataset (finaldata3)

# Label Encoder
from sklearn.preprocessing import LabelEncoder
# Creating instance of labelencoder
labelencoder = LabelEncoder()

finaldata3['Profile'] = labelencoder.fit_transform(finaldata3['Profile'])

#lets remove the not required variables
finaldata3 = finaldata3.drop(['created', 'Date', 'retailer_names', 'Recency', 'Frequency', 'Revenue', 'Cluster'], axis=1)


#Input and Output Split
x = finaldata3.iloc[:,:-1]
y = finaldata3.iloc[:,-1]

# To check for counts of Target/Label column
finaldata3["Profile"].value_counts()

## plotting of Target Column
plt.hist(finaldata3.Profile)

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# Classification Model - output/target variable - Profile

# Creating RandomForestClassifier Model in ensemble

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100,
                               criterion="gini",
                               min_samples_split=5,
                               min_samples_leaf=3,
                               random_state=100)
model.fit(x_train, y_train)

## Evaluation of Test dat (Prediction)
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)

result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)

result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2) 

# Train Data Accuracy
accuracy_score(y_train, model.predict(x_train)) 


# saving the model
# importing the model
import pickle

pickle.dump(model, open('modelbest.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('modelbest.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(finaldata3.iloc[:,:12])
list_value

print(model.predict(list_value))


finaldata3.columns
