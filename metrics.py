import metrics
import matplotlib.pyplot as plt
import seaborn as sn

#[true, predicted]
def recall(confusion_matrix, num_classes=2):
    if num_classes == 2:
        tp = confusion_matrix[1, 1]
        fn = confusion_matrix[1, 0]

        return tp/(tp + fn)
    else: #3 class
        tp = confusion_matrix[1, 1]
        fn = confusion_matrix[1, 0] + confusion_matrix[1, 2]

        return tp/(tp + fn)


def precision(confusion_matrix, num_classes=2):
    if num_classes == 2:
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1] 
        return tp/(tp + fp)
    else: #4 class
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1] + confusion_matrix[2, 1] 
        return tp/(tp + fp)

def f1(confusion_matrix, num_classes=2):
    precision = metrics.precision(confusion_matrix, num_classes)
    recall = metrics.recall(confusion_matrix, num_classes)
    return (2 * precision * recall) / (precision + recall);
    

def show_confusion(confusion_matrix):
    sn.set(font_scale=1.4)
    ax = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 16})
    ax.set(xlabel='Predicted Class', ylabel='True Class')
    ticklabels = ['Normal', 'Inf Covid', 'Inf Non-Covid']
    ax.set_xticklabels(ticklabels, size=8)
    ax.set_yticklabels(ticklabels, size=8)
    plt.show()
