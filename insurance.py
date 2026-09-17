"""Reversible feature transformations are a math exercise, not encryption."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=['gender','age','income','family_members']
RENAME={'Gender':'gender','Age':'age','Salary':'income','Family members':'family_members','Insurance benefits':'insurance_benefits'}

def fit_ols(X,y):
    X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=float)
    design=np.column_stack([np.ones(len(X)),X])
    return np.linalg.lstsq(design,y,rcond=None)[0]

def predict_ols(X,weights):
    return np.column_stack([np.ones(len(X)),np.asarray(X,dtype=float)])@weights

def verify_invariance(X_train,X_test,y_train,P):
    """A full-rank training design and invertible P preserve OLS predictions.

    Rank-deficient designs may have equal fitted training values but different
    minimum-norm extrapolations; reject that case for this held-out proof.
    """
    X_train,X_test,P=[np.asarray(x,dtype=float) for x in (X_train,X_test,P)]
    if P.shape!=(X_train.shape[1],X_train.shape[1]) or np.linalg.matrix_rank(P)<len(P):
        raise ValueError('P must be square and invertible')
    design=np.column_stack([np.ones(len(X_train)),X_train])
    if np.linalg.matrix_rank(design)<design.shape[1]: raise ValueError('Full column rank required')
    w=fit_ols(X_train,y_train)
    transformed=fit_ols(X_train@P,y_train)
    a=predict_ols(X_test,w); b=predict_ols(X_test@P,transformed)
    recovered=np.linalg.solve(P.T,(X_test@P).T).T
    delta=float(np.max(np.abs(a-b)))
    return {'prediction_max_absolute_difference':delta,
            'predictions_equal_at_1e_8':bool(np.allclose(a,b,rtol=1e-8,atol=1e-8)),
            'recovery_max_absolute_difference':float(np.max(np.abs(recovered-X_test))),
            'condition_number_P':float(np.linalg.cond(P))}

def evaluate_original(data):
    """Tune k on training CV; report one held-out classification/regression test."""
    df=data.rename(columns=RENAME).copy()
    cols=FEATURES+['insurance_benefits']
    if set(cols)-set(df): raise ValueError(f'Expected columns: {cols}')
    numeric=df[cols].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(numeric).all().all() or (numeric<0).any().any(): raise ValueError('Invalid or missing values')
    # Identical rows must not cross the train/test boundary.
    removed=int(numeric.duplicated().sum()); numeric=numeric.drop_duplicates()
    X=numeric[FEATURES]; y=numeric['insurance_benefits']; target=y.gt(0).astype(int)
    if target.value_counts().min()<10 or target.nunique()!=2: raise ValueError('Insufficient observations in each class')
    train,test=train_test_split(np.arange(len(X)),test_size=0.3,random_state=12345,stratify=target)
    pipeline=Pipeline([('scale',StandardScaler()),('knn',KNeighborsClassifier())])
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
    smallest_fold=len(train)-int(np.ceil(len(train)/5))
    search=GridSearchCV(pipeline,{'knn__n_neighbors':[k for k in [1,3,5,7,9,11] if k<=smallest_fold]},scoring='f1',cv=cv)
    search.fit(X.iloc[train],target.iloc[train])
    baseline=DummyClassifier(strategy='most_frequent').fit(X.iloc[train],target.iloc[train])
    reg=Pipeline([('scale',StandardScaler()),('ols',LinearRegression())]).fit(X.iloc[train],y.iloc[train])
    regbase=DummyRegressor(strategy='mean').fit(X.iloc[train],y.iloc[train])
    pred=reg.predict(X.iloc[test]); base_pred=regbase.predict(X.iloc[test])
    # Scale on train before the algebra experiment to reduce conditioning issues.
    scaler=StandardScaler().fit(X.iloc[train])
    train_x=scaler.transform(X.iloc[train]); test_x=scaler.transform(X.iloc[test])
    rng=np.random.default_rng(42)
    Q,_=np.linalg.qr(rng.normal(size=(X.shape[1],X.shape[1])))
    P=Q@np.diag([0.7,1.1,1.8,2.3])
    proof=verify_invariance(train_x,test_x,y.iloc[train],P)
    return {'dataset':'user-supplied insurance data','rows_after_deduplication':len(X),'exact_duplicates_removed':removed,
        'train_rows':len(train),'test_rows':len(test),'selected_k':search.best_params_['knn__n_neighbors'],
        'training_cv_f1':float(search.best_score_),
        'test_f1':float(f1_score(target.iloc[test],search.predict(X.iloc[test]))),
        'baseline_test_f1':float(f1_score(target.iloc[test],baseline.predict(X.iloc[test]),zero_division=0)),
        'test_rmse':float(np.sqrt(mean_squared_error(y.iloc[test],pred))),
        'baseline_test_rmse':float(np.sqrt(mean_squared_error(y.iloc[test],base_pred))),
        'test_r2':float(r2_score(y.iloc[test],pred)),'transformation_check':proof}

def demo():
    rng=np.random.default_rng(42)
    X=rng.normal(size=(250,4)); y=2+X@np.array([1.2,-0.5,0.8,2.0])+rng.normal(0,0.15,250)
    Q,_=np.linalg.qr(rng.normal(size=(4,4)))
    P=Q@np.diag([0.7,1.1,1.8,2.3])
    result=verify_invariance(X[:180],X[180:],y[:180],P)
    return {'dataset':'synthetic numerical proof','train_rows':180,'test_rows':70,**result}

if __name__=='__main__': print(json.dumps(demo(),indent=2))
