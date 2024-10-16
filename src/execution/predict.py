import argparse


def run_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset', type=int)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-ph', '--phase_file', type=str)
    parser.add_argument('-pm', '--phase_mask', type=str)
    parser.add_argument('-t1', '--t1_file', type=str)
    parser.add_argument('-tm', '--t1_mask', type=str)
    parser.add_argument('-tm', '--t1_mask', type=str)
    parser.add_argument('-TE', type=float)
    parser.add_argument('-B0', type=float, default=3)
    parser.add_argument('-ori', '--B0_vector', type=str, default='[0,0,1]')
    parser.add_argument('-to', '--to_file', type=str)

    args = parser.parse_args()

    print(args.phase_file, args.phase_mask, args.t1_file, args.phase_mask, args.TE, args.B0, eval(args.B0_vector))


if __name__ == '__main__':
    run_predict()
