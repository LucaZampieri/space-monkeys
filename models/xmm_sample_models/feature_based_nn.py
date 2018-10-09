import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras import layers, models, optimizers, regularizers, callbacks
import keras.backend as K
from sklearn.decomposition import PCA
from handyTools import stats
import load_xmm_data as xmm


class History(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.loss = []
        self.acc = []
        self.val_acc = []
        self.val_loss = []
        self.logloss = []
        self.val_logloss = []

        # initialize stats plot
        self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(3, 1)

        self.line_loss, = self.ax0.plot(self.loss, '-k', label='Training')
        self.line_val_loss, = self.ax0.plot(self.val_loss, '-r', label='Validation')
        self.ax0.set_ylabel('Loss')
        self.ax0.grid(True, which='both')

        self.line_acc, = self.ax1.plot(self.acc, '-k', label='Training')
        self.line_val_acc, = self.ax1.plot(self.val_acc, '-r', label='Validation')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.grid(True, which='both')

        self.line_logloss, = self.ax2.plot(self.logloss, '-k', label='Training')
        self.line_val_logloss, = self.ax2.plot(self.val_logloss, '-r', label='Validation')
        self.ax2.legend()
        self.ax2.set_ylabel('Log loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.grid(True, which='both')

    def on_epoch_end(self, batch, logs=None):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.logloss.append(logs.get('log_loss'))
        self.val_logloss.append(logs.get('val_log_loss'))

        # Update plot
        n = len(self.loss)

        self.line_loss.set_data((range(n), self.loss))
        self.line_val_loss.set_data((range(n), self.val_loss))
        self.line_acc.set_data((range(n), self.acc))
        self.line_val_acc.set_data((range(n), self.val_acc))
        self.line_logloss.set_data((range(n), self.logloss))
        self.line_val_logloss.set_data((range(n), self.val_logloss))

        self.ax0.set_xlim(xmin=0, xmax=n)
        try:
            self.ax0.set_ylim(ymin=0, ymax=max(max(self.loss), max(self.val_loss)))
        except ValueError:
            self.ax0.set_ylim(ymax=1)

        self.ax1.set_xlim(xmin=0, xmax=n)
        try:
            self.ax1.set_ylim(ymin=0, ymax=max(max(self.acc), max(self.val_acc)))
        except ValueError:
            self.ax1.set_ylim(ymax=1)

        self.ax2.set_xlim(xmin=0, xmax=n)
        try:
            self.ax2.set_ylim(ymin=0, ymax=max(max(self.logloss), max(self.val_logloss)))
        except ValueError:
            self.ax2.set_ylim(ymax=1)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def log_loss(y_true, y_pred):
    y = K.clip(y_pred, 1e-15, 1)

    return - K.sum(y_true * K.log(y))


def log_loss_weighted(weights):

    weights = K.variable(value=np.array([weights]))
    weights = K.transpose(weights)

    def loss(y_true, y_pred):

        y = K.clip(y_pred, 1e-15, 1 - 1e-15)
        return - K.sum(K.dot(y_true * K.log(y), weights)) / K.sum(weights)

    return loss



with open('features.pkl', 'rb') as f:
    feature_names, features, labels = pickle.load(f)

features = features[:,-4:]  # Disregard the hardness ratios
feature_names = feature_names[-4:]  # Disregard the hardness ratios

features = np.delete(features, (labels[:,3] == 1).nonzero(), axis=0)
labels = np.delete(labels, (labels[:,3] == 1).nonzero(), axis=0)
labels = np.delete(labels, [3], axis=1)

# Clean up the NaNs & Infs mess...
features[(np.isnan(features) | np.isinf(features)).nonzero()] = 0
features[np.abs(features) > 1e4] = 1e4 * np.sign(features[np.abs(features) > 1e4])

# Normalize the data
features = features - np.mean(features, axis=0)
# features = features / np.std(features, axis=0)
features = features / np.max(np.abs(features), axis=0)

# Dimensionality reduction: PCA
pca = PCA(
    # n_components=20,
    copy=True,
    whiten=False,
    svd_solver='auto'
)

# features_pca = pca.fit_transform(features)
features_pca = features

class_weights = {}
cls_w = labels.shape[0] / np.sum(labels, axis=0)
cls_w /= np.min(cls_w)
# cls_w = np.log(cls_w) + 1  # Log normalising the class weights

for k, l in enumerate(cls_w):
    class_weights[k] = l

print(class_weights)

# Split data into training and test sets
ind = np.random.choice(range(features.shape[0]), 1024, replace=False)

v_data = features_pca[ind,:]  # Validation
v_labels = labels[ind,:]

t_data = np.delete(features_pca, ind, axis=0)  # Training
t_labels = np.delete(labels, ind, axis=0)

n_classes = t_labels.shape[-1]
# n_batch = t_data.shape[0]
n_batch = 64

dropout = 0.5
reg = regularizers.l2(1e-4)
# reg = None

ins = layers.Input(shape=features_pca.shape[1:])

x = layers.Dense(
    1024,
    activation='relu',
    use_bias=True,
    kernel_regularizer=reg
)(ins)

x = layers.Dropout(dropout)(x)

x = layers.Dense(
    1024,
    activation='relu',
    use_bias=True,
    kernel_regularizer=reg
)(x)

x = layers.Dropout(dropout)(x)

# x = layers.Dense(
#     1024,
#     activation='relu',
#     use_bias=True,
#     kernel_regularizer=reg
# )(x)
#
# x = layers.Dropout(dropout)(x)

# x = layers.Dense(
#     64,
#     activation='relu',
#     use_bias=True,
#     kernel_regularizer=reg
# )(x)
#
# x = layers.Dropout(dropout)(x)

out = layers.Dense(
    n_classes,
    activation='softmax',
    use_bias=True,
    kernel_regularizer=reg
)(x)


net = models.Model(ins, out)
net.compile(
    optimizer='adamax',
    # loss='categorical_crossentropy',
    loss=log_loss_weighted(cls_w),
    metrics=['accuracy', log_loss]
)

hist = History()
net.fit(
    t_data, t_labels,
    epochs=1024,
    batch_size=n_batch,
    callbacks=[
        # callbacks.EarlyStopping(patience=8),
        hist,
        callbacks.ModelCheckpoint(
            './temp_xmm_features_weights_1.hdf5',
            save_best_only=True,
            save_weights_only=True,
            monitor='val_acc'
        ),
    ],
    validation_data=(v_data, v_labels),
    shuffle=True,
    # class_weight=class_weights
)

# Test on validation data
predicted = net.predict(v_data)

v_cls = np.argmax(v_labels, axis=1)
p_cls = np.argmax(predicted, axis=1)

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

cm, fig = stats.plot_confusion_matrix(v_cls, p_cls, class_names=cls, normalize=False)
plt.title('Confusion matrix on validation data')

cpm, fig2 = stats.plot_cpm(v_cls, predicted, class_names=cls)
plt.title('Conditional Probability Matrix on validation data')

# Calculate accuracy
n = cm.shape[0]
acc = np.sum(cm[range(n), range(n)]) / np.sum(cm)
logloss = - np.sum(v_labels * np.log(np.clip(predicted, 1e-15, 1)))
print('Accuracy on validation data: {:.3%}'.format(acc))
print('Log loss on validation data: {:.5f}'.format(logloss))

# Uncertainty

plt.figure()
plt.fill_between(
    *stats.hist(np.max(predicted, axis=1)),
    color='k', alpha=0.5, edgecolor=None
)
plt.axvline(1 / n_classes, color='k', linestyle='--')
plt.ylim(ymin=0)
plt.xlim(xmax=1)
plt.xlabel('Classification probability (top class)')
plt.title('Classification probability histogram')

# Calibration

probs = np.max(predicted, axis=1)
probs.sort()
prob_bins = probs[0::64]
delta = np.diff(prob_bins)
rtp = []
for p in range(len(prob_bins) - 1):
    i = np.nonzero(
        (np.max(predicted, axis=1) > prob_bins[p]) &
        (np.max(predicted, axis=1) <= prob_bins[p + 1])
    )[0]

    ntp = np.sum(p_cls[i] == v_cls[i])
    rtp.append(ntp / len(i))

rtp = np.array(rtp)
prob2 = 0.5 * (prob_bins[1:] + prob_bins[:-1])

plt.figure()
plt.plot([0, 1], [0, 1], '--k')
plt.errorbar(prob2, rtp, xerr=0.5 * delta, fmt='or')
plt.ylim(ymin=0, ymax=1)
plt.xlim(xmin=0, xmax=1)
plt.grid()
plt.title('Calibration test')
plt.xlabel('Classification probability')
plt.ylabel('True Positive rate')
