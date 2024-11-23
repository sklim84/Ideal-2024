import argparse

parser = argparse.ArgumentParser()

# Commons
parser.add_argument('--method', type=str, default='fedsgd')
parser.add_argument('--model_name', type=str, default='ctgan')
parser.add_argument('--num_samples_org', type=int, default=300)
parser.add_argument('--num_samples_syn', type=int, default=300)
parser.add_argument('--org_data_path', type=str, default='./datasets/DATOP_HF_TRANS_100_102_104_iid.csv')
parser.add_argument('--syn_data_path', type=str, default='./datasets_syn/')
parser.add_argument('--results_path', type=str, default='./results/eval_results.csv')
parser.add_argument('--seed', type=int, default=2024)

# CTGAN
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=500)

parser.add_argument('--emb_dim', type=int, default=16)
parser.add_argument('--gen_dim', type=int, default=16)
parser.add_argument('--dis_dim', type=int, default=16)

def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
