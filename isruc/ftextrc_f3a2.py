from scipy.io import loadmat
from scipy.signal import welch
import matplotlib as plt
import numpy as np

file = 'subject001/subject01.mat'
data = loadmat(file)

epoch = data['F3_A2'][0, :]

def hjorth_params(epoch):
    first_deriv = np.diff(epoch)
    second_deriv = np.diff(first_deriv)

    activity = np.var(epoch)
    mobility = np.sqrt(np.var(first_deriv) / activity)
    complexity = np.sqrt(np.var(second_deriv) / np.var(first_deriv)) / mobility
    return activity, mobility, complexity

feature_list = []

for epoch in data['F3_A2']:
    #psd features
    frequencies, psd = welch(epoch, fs = 200, nperseg = 200*3) #wtf is sampling rate guys
    #sum powe rin freq bands (Hz ranges)
    delta_power = np.sum(psd[(frequencies >= 0.5) & (frequencies < 4)])
    theta_power = np.sum(psd[(frequencies >= 4) & (frequencies < 8)])
    alpha_power = np.sum(psd[(frequencies >= 8) & (frequencies < 13)])
    beta_power  = np.sum(psd[(frequencies >= 13) & (frequencies < 30)])

    activity, mobility, complexity = hjorth_params(epoch)

    # 3. Variance
    variance = np.var(epoch)

    # 4. Collect Features
    features = [delta_power, theta_power, alpha_power, beta_power, activity, mobility, complexity, variance]
    feature_list.append(features)

X = np.array(feature_list)
print(X)
print(X.shape)