import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import TransformerMixin

class Encoder(TransformerMixin):
    def __init__(self, columns=None, thresh=0, sep='/', drop=True, as_int=True, drop_one=True):
        self.columns = columns
        self.thresh = thresh
        self.encoding = None
        self.sep = sep
        self.drop = drop
        self.as_int = as_int
        self.drop_one = drop_one
    
    def fit(self, X, y=None):
        c = self.columns
        if c is None: c = self.numeric_columns(X, inv=True).columns
        self.encoding = dict()
        if isinstance(self.thresh, float):
            thresh = int(len(X) * self.thresh)
        else: thresh = self.thresh
        for col in c:
            e = self._encode_column(X[col], thresh)
            if len(e) > 0: self.encoding[col] = e
        return self

    def transform(self, X, y=None, inplace=False):
        if not inplace: X = X.copy()
        for col in self.encoding:
            self._apply_encoding(X, col)
        return X

    def _encode_column(self, s, thresh):
        output = []
        seen = defaultdict(int)
        for val, n in s.value_counts().iteritems():
            if self.sep in val:
                for v in val.split(self.sep):
                    if s.apply(lambda x: str(v) in str(x)).sum() >= thresh and (v not in seen):
                            output.append(v)
                    seen[v] += 1
            else:
                if n >= thresh and val not in seen:
                    output.append(val)
                seen[val] += 1

        return output, seen

    def _encode_column(self, s, thresh):
        seen = defaultdict(int)
        for val, n in s.value_counts().iteritems():
            if self.sep in val:
                for v in val.split(self.sep):
                    seen[v] += n
            else:
                seen[val] += n
        output = [v for v, cnt in seen.items() if cnt >= thresh]
        if self.drop_one and len(output) == len(seen):
            output.remove(max(seen, key=seen.__getitem__))
        return output

    def _apply_encoding(self, X, col):
        s = X[col]
        for val in self.encoding[col]:
            has_sep = s.apply(lambda x: self.sep in str(x))
            X[f'{col}_{val}'] = np.where(has_sep, 
                s.apply(lambda x: str(val) in str(x)), 
                (s.astype(type(val))==val))
            if self.as_int:
                X[f'{col}_{val}'] = X[f'{col}_{val}'].astype(int)
        if self.drop: X.drop(col, axis=1, inplace=True)

    @staticmethod
    def numeric_columns(X, inv=False):
        numeric_cols = [c for c,b in X.apply(Encoder.can_cast).items() if b]
        return X.drop(columns=numeric_cols) if inv else X.loc[:, numeric_cols]

    @staticmethod
    def can_cast(x, dtypes=(np.int64, np.float64, np.datetime64)):
        for dtype in dtypes:
            try:
                dtype(x)
                return True
            except (ValueError, TypeError):
                pass
        return False