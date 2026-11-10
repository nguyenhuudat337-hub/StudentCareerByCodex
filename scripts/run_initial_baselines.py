from career_ml.data import load_data,prepare_xy,split_data
from career_ml.initial import run_initial_baselines
if __name__=="__main__":
    X,y,_=prepare_xy(load_data())
    X_train,_,y_train,_=split_data(X,y)
    run_initial_baselines(X_train,y_train)
