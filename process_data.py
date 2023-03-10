from utils.data_preprocess import h36m_train_extract
from utils.data_preprocess import internet_data_extract
import config
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, required=True, choices=['3dpw', '3dhp', 'h36m', 'internet'],
# help='process which dataset?')
parser.add_argument('--dataset', type=str, default='h36m')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'h36m':
        # h36m_train_extract(config.H36M_ROOT, training_split=False, extract_img=False)
        h36m_train_extract(config.H36M_ROOT, training_split=True)
    elif args.dataset == 'internet':
        internet_data_extract(config.InternetData_ROOT)
    else:
        print('Not implemented.')
