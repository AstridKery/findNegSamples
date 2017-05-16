import matplotlib.pyplot as plt
import numpy as np

# load data
data = np.load("AllWords_NumpyResults.npy")
rows, columns = data.shape

# sort per row
data = np.sort(data, axis=1)

# processed data
varianceData = []

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
    variance = np.var(wordData)
    varianceData.append(variance)

plt.xlabel("Document")
plt.ylabel("TF-IDF")
plt.title("Sorted TF-IDF values for different words")
plt.savefig("TF-IDF")

plt.figure()
plt.hist(varianceData, 50)
plt.xlabel("Variance")
plt.ylabel("Variance")
plt.title("Variance for word plots")
plt.savefig("variance")
