from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

class PreprocessorLog1(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        pass
    
    def transform(self, X, *_):
        new_X = self._preprocess_input(X)
        return new_X
    
    def fit(self, X, *_):
        return self
    
    # courtesy of pham minh
    def _preprocess_input(self, X):
        df_X = pd.DataFrame(X, columns=self.columns)
        # Turn age feature into log(age)
        if ("log10(age+1)" not in df_X.columns) and ("age" in df_X.columns):
            df_X["log10(age+1)"] = df_X["age"].transform(lambda x: np.log10(x + 1))
            df_X.drop("age", axis = 1, inplace = True)
        # Drop unnecessary features
        dropped_features = [
            "has_secondary_use",
            "has_secondary_use_agriculture",
            "has_secondary_use_hotel",
            "has_secondary_use_rental",
            "has_secondary_use_institution",
            "has_secondary_use_school",
            "has_secondary_use_industry",
            "has_secondary_use_health_post",
            "has_secondary_use_gov_office",
            "has_secondary_use_use_police",
            "has_secondary_use_other"
        ]
        for feature in dropped_features:
            if feature in df_X.columns:
                df_X.drop(feature, axis = 1, inplace = True)
        # Turn all category features into multiple one-hot features
        df_X = pd.get_dummies(df_X)
        return df_X.values

class PreprocessorXGB3(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        pass
    
    def transform(self, X, *_):
        new_X = self._preprocess_input(X)
        return new_X
    
    def fit(self, X, *_):
        return self
    
    def _preprocess_input(self, X):
        df_X = pd.DataFrame(X, columns=self.columns)
        # Commented means not good
        #     df['area_height_ratio'] = df['area_percentage'] / df['height_percentage']
        df_X['count_floors_height_ratio'] = df_X['count_floors_pre_eq'] / df_X['height_percentage']
        #     df['age_count_floors_pre_eq_ratio'] = df['age'] / df['count_floors_pre_eq']
        df_X = pd.get_dummies(df_X)
        return df_X.values

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred) - 1]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax