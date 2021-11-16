from sklearn.datasets import load_boston
import pandas as pd
from autoML import AutoML

if __name__ == '__main__':
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['price'] = boston.target
    print(df)
    automl = AutoML(df, 'price', min_x_y_correlation_rate=0.25, n_features_threshold=0.8)
    print(automl.getResults())
