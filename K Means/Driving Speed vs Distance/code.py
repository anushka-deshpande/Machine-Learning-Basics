import pandas as pd
df = pd.read_csv('Dataset_Driver.csv')
print("Head:\n", df.head())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_scaled = sc.fit_transform(df)
print("\nScaled_df: \n", df_scaled[:5,])

from sklearn.cluster import KMeans
ssq=[]
for K in range(1,11):
    myModel = KMeans(n_clusters=K, random_state=123)
    result = myModel.fit(df_scaled)
    ssq.append(result.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,11), ssq, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster SSQ')
plt.title('Scree Plot')
plt.plot([4]*3000, range(1, 3001), ":")
plt.text(4.1, 3000, 'optimal number of culsters = 4')
plt.show()

myModel = KMeans(n_clusters=4, random_state=163)
result = myModel.fit(df_scaled)
print("\nResult Labels: \n", result.labels_)

predictions = result.predict(df_scaled)
print("\nPredictions: \n", predictions[:5])

plt.scatter(df_scaled[predictions == 0,0], df_scaled[predictions==0,1], s=50, c='lightgreen', marker='s', edgecolors='black', label='Cluster 1')
plt.scatter(df_scaled[predictions == 1,0], df_scaled[predictions==1,1], s=50, c='orange', marker='o', edgecolors='black', label='Cluster 2')
plt.scatter(df_scaled[predictions == 2,0], df_scaled[predictions==2,1], s=50, c='lightblue', marker='v', edgecolors='black', label='Cluster 3')
plt.scatter(df_scaled[predictions == 3,0], df_scaled[predictions==3,1], s=50, c='yellow', marker='s', edgecolors='black', label='Cluster 4')
plt.scatter(result.cluster_centers_[:,0], result.cluster_centers_[:,1], s=250, c='red', marker='*',edgecolors='black',label='centroids')


plt.legend(scatterpoints=1)
plt.xlabel("Driving Distance")
plt.ylabel("Driving Speed")
plt.title("Clustering Output")
plt.show()

import numpy as np
predictions_relabelled = np.where(predictions==0, 'A', np.where(predictions==1, 'B', np.where(predictions==2, 'C', 'D')))
df['category'] = pd.Series(predictions_relabelled, index=df.index)
df.index.name = 'Number'
print("\nNew Dataframe head: \n", pd.DataFrame(df).head())
print("\nNew Dataframe tail: \n", pd.DataFrame(df).tail())

df.to_csv("driver_segmentation_output.csv",index=False)

