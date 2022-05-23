import sys
import torch

from dasbe.clustering_dynamics_for_static_dataset import *


if __name__ == "__main__":
    net_name = sys.argv[1]
    dataset_name = sys.argv[2]

    net = torch.load('./dasbe/tmp_net/' + net_name) 

    _, multi, _, _, multi_label, _ = gain_dataset("./dasbe/tmp_data", dataset_name)

    # choose data index
    # i = np.random.randint(0,6000)
    i = 1109
    print('CHOOSE DATA ID: ', i)

    spk, enc = clustering(np.array(multi[i], dtype=np.float32), net, multi_label[i], s_refractory=9, a_refractory=9, hidden_size=[20, 20])
       
    labeled_synchrony_measure(spk, multi_label[i])
    
    fMRI_measure(spk, multi_label[i])

    pred_low, _ = k_means_var(spk, multi_label[i], K=[4], back=10, smooth=1)

    pred_high, _ = k_means_var(enc, multi_label[i], K=[4], back=10, smooth=1)
    
    decode_with_kmeanMask(net, enc, pred_high, multi[i])
    
    AMI_score = evaluate_grouping(multi_label[i], pred_low)

