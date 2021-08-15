from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

# y_true = [0,0,1,2,1,2,0,2,2,0,1,1]
# y_pred = [1,0,1,2,1,0,0,2,2,0,1,1]
# labels = ['0', '1', '2']
# labels = [0, 1, 2]
# cm = confusion_matrix(y_true, y_pred)
# sns.set()
# sns.heatmap
# plt.figure()
# plt.

def make_confusion_matrix(y_true,y_pred,labels,normalize=False, vis=True, method='confusion matrix'):
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred,labels=labels)
    if normalize:
        np.set_printoptions(precision=2)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    if vis:
        plt.figure(figsize=(12, 8), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='white' if c > cm.max()/2 else 'black', fontsize=10, va='center', ha='center')
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plot_confusion_matrix(cm, labels,title=method)
        plt.savefig('confusion_matrix.png', format='png')
        plt.show()

def plot_confusion_matrix(cm,labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# make_confusion_matrix(y_true,y_pred,labels,normalize=False)
#
# np.set_printoptions(precision=2)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# cm_normalized1 = cm_normalized
# print (cm_normalized)
# plt.figure(figsize=(12, 8), dpi=120)
#
# ind_array = np.arange(len(labels))
# x, y = np.meshgrid(ind_array, ind_array)
#
# for x_val, y_val in zip(x.flatten(), y.flatten()):
#     c = cm_normalized[y_val][x_val]
#     if c > 0.01:
#         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# # offset the tick
# plt.gca().set_xticks(tick_marks, minor=True)
# plt.gca().set_yticks(tick_marks, minor=True)
# plt.gca().xaxis.set_ticks_position('none')
# plt.gca().yaxis.set_ticks_position('none')
# plt.grid(True, which='minor', linestyle='-')
# plt.gcf().subplots_adjust(bottom=0.15)
#
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# # show confusion matrix
# # plt.savefig('confusion_matrix.png', format='png')
# plt.show()