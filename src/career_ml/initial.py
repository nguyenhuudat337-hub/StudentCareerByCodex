"""Baseline mặc định của 3 thuật toán trên Train CV, CHƯA GridSearchCV."""
import time
import pandas as pd
from sklearn.model_selection import cross_validate
from .config import MODEL_NAMES,OUTPUT
from .modeling import make_pipeline,make_cv


def run_initial_baselines(X_train,y_train,output=OUTPUT,jobs=2):
    rows=[]
    for name in MODEL_NAMES:
        t=time.perf_counter()
        pipeline=make_pipeline(X_train,name,"None")
        scores=cross_validate(pipeline,X_train,y_train,cv=make_cv(),
                              scoring={"f1_macro":"f1_macro","accuracy":"accuracy"},n_jobs=jobs)
        row={"model":name,"fs":"None","gridsearch":False,
             "cv_f1_macro":float(scores["test_f1_macro"].mean()),
             "std_cv_f1_macro":float(scores["test_f1_macro"].std()),
             "cv_accuracy":float(scores["test_accuracy"].mean()),
             "seconds":round(time.perf_counter()-t,2)}
        rows.append(row)
        print(f"[Mặc định] {name}: F1 Macro CV={row['cv_f1_macro']:.4f}",flush=True)
    path=output / "tables" / "initial_baselines_cv.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    return rows
