import argparse
import platform
from scripts.trainer import NetTrainer

if platform.system().lower() == 'windows':
    pf = r'E:\Data\runtime\mQSM\mQSM_processed'
    rf = r'E:\Data\runtime\mQSM\mQSM_results'
else:
    pf = '/remote-home/hejj/Data/runtime/mQSM/mQSM_processed'
    rf = '/remote-home/hejj/Data/runtime/mQSM/mQSM_results'


def run_trainer():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', type=str, default=pf, help='Path to the processed data')
    parser.add_argument('-r', type=str, default=rf, help='Path to the result data')
    parser.add_argument('-f', type=int, default=0, help='Train fold')
    parser.add_argument('-D', type=int, default=3, help='Dataset ID')

    parser.add_argument('-b', type=int, default=2, help='Batch size')
    parser.add_argument('--c', action='store_true', help='Continue training')
    parser.add_argument('--v', action='store_true', help='Validation')
    parser.add_argument('-d', type=str, default='0', help='GPU ID')
    parser.add_argument('-e', type=int, default=3, help='Epochs')
    args = parser.parse_args()

    tr = NetTrainer(in_channels=2,
                    num_classes=37,
                    patch_size=[128, 128, 64],
                    batch_size=args.b,
                    dataset_id=args.D,
                    processed_folder=args.p,
                    result_folder=args.r,
                    fold=args.f,
                    go_on=args.c,
                    epochs=args.e,
                    device=args.d,
                    validation=args.v,
                    logger=print)
    tr.run()


if __name__ == '__main__':
    run_trainer()
