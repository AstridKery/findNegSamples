import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# load data
data = np.load("AllWords_NumpyResults.npy")
data = normalize(data)
rows, columns = data.shape

# sort per row
data = np.sort(data, axis=1)

# processed data
varianceData = []
meanData = []

# plot things
plt.figure()
for wordIndex in range(rows):
    # extract list to plot
    # reverse list before plotting
    # extract data for this word
    # add reflected data to make it look like a gaussian curve
    wordData = list(data[wordIndex,:])[::1] + list(data[wordIndex,:])[::-1]
    plt.plot(wordData)

    # compute mean and variance
    mean = np.mean(wordData)
    variance = np.var(wordData)
    meanData.append(mean)
    varianceData.append(variance)

# cluster based on variance
kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(varianceData).reshape(-1,1))
print sum(kmeans.labels_ == 0)
print sum(kmeans.labels_ == 1)
print sum(kmeans.labels_ == 2)

# plot things
plt.xlabel("Document")
plt.ylabel("TF-IDF")
plt.title("Sorted TF-IDF values for different words")
plt.savefig("TF-IDF")

plt.figure()
plt.hist(varianceData, 50)
plt.xlabel("Variance")
plt.ylabel("Number of Words")
plt.title("Variance for word plots")
plt.savefig("variance_histogram")

plt.figure()
plt.scatter(varianceData, varianceData, c=kmeans.labels_)
plt.xlabel("Variance")
plt.ylabel("Variance")
cbar = plt.colorbar(label="Cluster labels")
plt.title("Variance for word plots")
plt.savefig("variance_scatter")

plt.figure()
plt.hist(meanData, 50)
plt.xlabel("Mean")
plt.ylabel("Number of Words")
plt.title("Mean for word plots")
plt.savefig("mean_histogram")

plt.figure()
plt.scatter(meanData, meanData)
plt.xlabel("Mean")
plt.ylabel("Mean")
plt.title("Mean for word plots")
plt.savefig("mean_scatter")

plt.figure()
plt.scatter(meanData, varianceData)
plt.xlabel("Mean")
plt.ylabel("Variance")
plt.title("Mean and variance for word plots")
plt.savefig("mean_variance_scatter")
