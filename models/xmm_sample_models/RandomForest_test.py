import numpy as np
import matplotlib.pyplot as plt
import pickle

from handyTools import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
# import load_xmm_data as xmm


with open('features.pkl', 'rb') as f:
    feature_names, features, labels = pickle.load(f)

# features = features[:,:-4]  # Disregard the hardness ratios
# feature_names = feature_names[:-4]  # Disregard the hardness ratios

# Clean up the NaNs & Infs mess...
features[(np.isnan(features) | np.isinf(features)).nonzero()] = 0

# Normalize the data
features[np.abs(features) > 1e4] = 1e4 * np.sign(features[np.abs(features) > 1e4])
features = features - np.mean(features, axis=0)
features = features / np.max(np.abs(features), axis=0)
# features = features / np.std(features, axis=0)
# features[:,19] = features[:,19] / np.max(np.abs(features[:,19]))

features = np.delete(features, (labels[:,3] == 1).nonzero(), axis=0)
labels = np.delete(labels, (labels[:,3] == 1).nonzero(), axis=0)
labels = np.delete(labels, [3], axis=1)

labels = np.argmax(labels, axis=1)

# Dimensionality reduction: PCA
pca = PCA(
    n_components=10,
    copy=True,
    whiten=False,
    svd_solver='auto'
)

features_pca = pca.fit_transform(features)

# Split data into training and test sets
ind = np.random.choice(range(features.shape[0]), 1024, replace=False)

v_data = features_pca[ind,:]  # Validation
v_labels = labels[ind]

t_data = np.delete(features_pca, ind, axis=0) # Training
t_labels = np.delete(labels, ind, axis=0)


# Create the Random Forest Classifier

rf = RandomForestClassifier(
    max_features='auto',
    class_weight='balanced',
    n_jobs=-1,
    n_estimators=100,
    criterion='gini'
)

# Train the classifier
rf.fit(t_data, t_labels)

# Test on validation data
p_cls = rf.predict(v_data)

cls = [
    'XRB',
    'CV',
    'GRB',
    # 'SSS',
    'Star',
    'Galaxy',
    'AGN',
    'ULX'
]

cm, fig = stats.plot_confusion_matrix(v_labels, p_cls, class_names=cls, normalize=False)
plt.title('Random Forest - validation data')

# Calculate accuracy
n = cm.shape[0]
acc = np.sum(cm[range(n), range(n)]) / np.sum(cm)
print('Accuracy on validation data: {:.3%}'.format(acc))


# Plot feature importance

i = np.argsort(rf.feature_importances_)

fig2 = plt.figure()
# plt.barh(range(len(feature_names)), rf.feature_importances_[i], color='k', alpha=0.5)
plt.barh(range(i.size), rf.feature_importances_[i], color='k', alpha=0.5)
plt.xlim(xmin=-0.5 * max(rf.feature_importances_))
plt.axis('off')
for x, t, name in zip(range(features_pca.shape[1]), rf.feature_importances_[i], i):
    plt.text(t + 0.01 * max(rf.feature_importances_), x,
             '{:.1%}'.format(t),  verticalalignment='center')
    plt.text(-0.01 * max(rf.feature_importances_), x,
             name, verticalalignment='center', horizontalalignment='right')
plt.title('Feature importance')

# plt.tight_layout()
