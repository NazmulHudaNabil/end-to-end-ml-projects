import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Module 1: Text Cleaners
class LevyCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Levy'] = X['Levy'].fillna({"-":np.nan})
        return X



class EngineVolumeCleaner(BaseEstimator, TransformerMixin):
    """Strip 'Turbo' keyword from Engine volume column."""
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = X.copy()
        X['Engine volume'] = (
            X['Engine volume']
            .str.replace(r'\bTurbo\b', '', regex=True)
            .str.strip()
        )
        return X




class MileageCleaner(BaseEstimator, TransformerMixin):
    """Strip 'km' unit label from Mileage column."""
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = X.copy()
        X['Mileage'] = (
            X['Mileage']
            .str.replace(r'\bkm\b', '', regex=True)
            .str.strip()
        )
        return X
    


# Module 2: Type Converter
class NumericConverter(BaseEstimator, TransformerMixin):
    """Convert specified columns to numeric data type."""
 
    def __init__(self, columns):
        self.columns = columns
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X
    
# Module 3: Missing Value Imputer
class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.medians_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.medians_[col] = X[col].median()

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.medians_[col])
        return X
    

# Module 4: Dupplicate reomover
class DuplicateRemover(BaseEstimator, TransformerMixin):
    """Drop duplicate rows and reset the index."""
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X):
        return X.drop_duplicates(keep='first').reset_index(drop=True)
    



# Module 5: IQROutlierFilter
class IQROutlierFilter(BaseEstimator, TransformerMixin):
    """Remove rows where any specified column falls outside 1.5×IQR bounds."""
 
    def __init__(self, columns):
        self.columns = columns
        self.bounds_ = {}
 
    def fit(self, X, y=None):
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return self
 
    def transform(self, X):
        X = X.copy()
        mask = np.ones(len(X), dtype=bool)
        for col in self.columns:
            lower, upper = self.bounds_[col]
            mask &= X[col].between(lower, upper)
        return X[mask].reset_index(drop=True)
    

# Assemble the Pipeline
NUMERICAL_COLS = ['Mileage', 'Engine volume', 'Levy']

preprocessing_pipeline = Pipeline(steps=[
    ('levy_cleaner', LevyCleaner()),
    ('engine_volume_cleaner', EngineVolumeCleaner()),
    ('mileage_cleaner', MileageCleaner()),
    ('numeric_converter', NumericConverter(columns=NUMERICAL_COLS)),
    ('median_imputer', MedianImputer(columns=NUMERICAL_COLS)),
    ('duplicate_remover', DuplicateRemover()),
    ('iqr_outlier_filter', IQROutlierFilter(columns=NUMERICAL_COLS))
])
