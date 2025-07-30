#-----to see actaul files
import numpy as np
import pandas as pd

features = np.load('../results/batches/features_batch_1.npy')
labels = np.load('../results/batches/labels_batch_1.npy')

features2 = np.load('../results/batches/features_batch_2.npy')
labels2 = np.load('../results/batches/labels_batch_2.npy')

features3 = np.load('../results/batches/features_batch_3.npy')
labels3 = np.load('../results/batches/labels_batch_3.npy')

features4 = np.load('../results/batches/features_batch_4.npy')
labels4 = np.load('../results/batches/labels_batch_4.npy')

# Print first 5 rows of features and labels
df = pd.DataFrame(features, columns=['Delta', 'Theta', 'Alpha', 'Beta', 'Activity', 'Mobility', 'Complexity', 'Variance'])
df['Label'] = labels

df2 = pd.DataFrame(features2, columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Activity', 'Mobility', 'Complexity', 'Variance'])
df2['label'] = labels2

df3 = pd.DataFrame(features3, columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Activity', 'Mobility', 'Complexity', 'Variance'])
df3['label'] = labels3

df4 = pd.DataFrame(features4, columns = ['Delta', 'Theta', 'Alpha', 'Beta', 'Activity', 'Mobility', 'Complexity', 'Variance'])
df4['label'] = labels4

print('Features shape:', features.shape)
print('Labels shape:', labels.shape)
print(df.head(600))

print('features2 shape:', features2.shape)
print('labels2 shape: ', labels2.shape)
print(df2.head(600))

print('features3 shape:', features3.shape)
print('labels3 shape: ', labels3.shape)
print(df3.head(600))

print('features4 shape:', features4.shape)
print('labels4 shape: ', labels4.shape)
print(df4.head(600))