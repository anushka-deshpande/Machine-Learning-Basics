# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# acquire data
df = pd.read_csv("Dataset_Customer.csv")

# print head
print("Head: \n" , df.head())

# print shape
print("\nShape: \n", df.shape)

# gets stats info
print("\nStatistical Information: \n", df.describe())

# data preprocessing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_scaled = sc.fit_transform(df)
print("\nScaled Dataframe: \n", df_scaled)

# training the model
from sklearn.cluster import KMeans
# initially value of K is not determined
ssq = []
# array of sum of squared distances

for K in range(1,11):
    my_model = KMeans(n_clusters=K, random_state=123);
    result = my_model.fit(df_scaled)
    ssq.append(my_model.inertia_)
    # inertia is the error in sum of squared distances

# generate plot with matplotlib
plt.plot(range(1,11), ssq, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Within-cluster SSQ")
plt.title("SSQ Plot")
plt.show()

# from model we can see that K=3 is most optimal
my_model = KMeans(n_clusters=3, random_state=123)
result = my_model.fit(df_scaled)

# testing the model
predictions = result.predict(df_scaled)
print("\nFirst 5 predictions: \n", predictions[:5])

# deploying the model
pred_new = list(result.predict([[9.3,5.36]]))
print("\nNew Prediction: \n", pred_new)

# plot data partitioned into clusters
# cluster, column0
# cluster, column1
# s=size
# c=color
plt.scatter(df_scaled[predictions==0,0], df_scaled[predictions==0,1], s=50, c='lightgreen',\
            marker='s', edgecolors='black', label='cluster 1')

plt.scatter(df_scaled[predictions==1,0], df_scaled[predictions==1,1], s=50, c='orange',\
            marker='o', edgecolors='black', label='cluster 2')

plt.scatter(df_scaled[predictions==2,0], df_scaled[predictions==2,1], s=50, c='lightblue',\
            marker='v', edgecolors='black', label='cluster 3')

plt.scatter(result.cluster_centers_[:,0], result.cluster_centers_[:,1], s=250, c='red',\
            marker='*', edgecolors='black', label='centroids')

plt.legend(scatterpoints=1)
plt.xlabel("Brand Loyalty Score")
plt.ylabel("Price Sensitivity Score")
plt.title("Clustering Output")
plt.show()

# relabel predictions
predictions_relabelled = np.where(predictions==0, "Value Conscious", np.where(predictions==1, "Brand Advocates", "Loyal to low cost"))
df['category'] = pd.Series(predictions_relabelled, index=df.index)
df.index.name = "Customer Number"

print("\nNew Dataframe: \n", pd.DataFrame(df).head())

# convert new df to csv
df.to_csv("marketing_segmentation_output.csv", index=False)
