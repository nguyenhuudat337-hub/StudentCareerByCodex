import argparse
from career_ml.explain import explain
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--samples",type=int,default=80)
    args=p.parse_args()
    explain(sample_size=args.samples)
