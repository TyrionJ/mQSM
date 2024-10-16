import argparse
import platform
from scripts.preprocessor import Processor

if platform.system().lower() == 'windows':
    pf = r'E:\Data\runtime\mQSM\mQSM_processed'
    rf = r'E:\Data\runtime\mQSM\mQSM_raw'
else:
    pf = '/remote-home/hejj/Data/runtime/mQSM/mQSM_processed'
    rf = '/remote-home/hejj/Data/runtime/mQSM/mQSM_raw'


def run_preprocess():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=pf)
    parser.add_argument('-r', type=str, default=rf)
    parser.add_argument('-D', type=int, default=3)
    args = parser.parse_args()

    tr = Processor(dataset_id=args.D, raw_folder=args.r, processed_folder=args.p)
    tr.run()


if __name__ == '__main__':
    run_preprocess()
