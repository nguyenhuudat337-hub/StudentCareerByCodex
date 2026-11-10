import numpy as np
from career_ml.config import EXCLUDE_COLUMNS
from career_ml.data import load_data,prepare_xy,split_data,audit_data
from career_ml.modeling import make_pipeline,feature_names_after_fs,param_grid
from career_ml.evaluation import validate_proba


def test_dataset_and_exclusions():
    df=load_data(); a=audit_data(df)
    assert len(df)==12000 and a["missing_total"]==0
    assert a["duplicate_rows"]==0 and len(a["target_counts"])==12
    X,y,encoder=prepare_xy(df)
    assert "recommended_career" not in X.columns
    assert all(col not in X.columns for col in EXCLUDE_COLUMNS)
    assert len(encoder.classes_)==12
    Xtr,Xte,ytr,yte=split_data(X,y)
    assert len(Xtr)==9600 and len(Xte)==2400
    assert set(Xtr.index).isdisjoint(set(Xte.index))


def test_grids_and_feature_selection():
    df=load_data();X,y,enc=prepare_xy(df)
    Xtr,_,ytr,_=split_data(X,y)
    Xsmall=Xtr.iloc[:144];ysmall=ytr.iloc[:144]
    for model_name in ["Logistic Regression","Random Forest","XGBoost"]:
        for kind in ["None","Filter","Wrapper","Embedded"]:
            grid=param_grid(model_name,kind,"quick")
            assert len(grid)>0
            pipe=make_pipeline(Xsmall,model_name,kind)
            pipe.fit(Xsmall,ysmall)
            p=validate_proba(pipe,Xsmall.iloc[:3],len(enc.classes_))
            assert p.shape==(3,12)
            assert len(feature_names_after_fs(pipe))>=1
