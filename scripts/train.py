import argparse
from career_ml.experiment import run_experiment

def main():
    p = argparse.ArgumentParser(
        description="Phân loại nghề nghiệp với 3 mô hình và 12 cấu hình thực nghiệm"
    )
    p.add_argument("--profile",default="quick",choices=["quick","standard","full"])
    p.add_argument("--jobs",type=int,default=2,help="Số tiến trình song song trong CV")
    p.add_argument("--limit",type=int,default=None,help="DEBUG: chỉ chạy N mô hình, không Test")
    p.add_argument("--debug-rows",type=int,default=None,help="DEBUG: lấy mẫu cân bằng; không Test")
    p.add_argument("--compare-calibration",action="store_true",help="Thử sigmoid trên validation thuộc Train")
    args=p.parse_args()
    run_experiment(preset=args.profile,jobs=args.jobs,limit=args.limit,
                   debug_rows=args.debug_rows,compare_calibration=args.compare_calibration)
if __name__=="__main__":
    main()
