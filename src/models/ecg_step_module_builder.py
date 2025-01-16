import sys

sys.path.insert(1, "/Users/haoyangchen/Desktop/ChenHaoyang/coding/python/ECG_XAI")

from ecg_step_module import EcgModule

if __name__ == '__main__':
    data = {'Name': 'AIB',
            'focused leads': "['V1', 'V2', 'V5', 'V6']",
            'obj_feat_names': "['LQRS', 'LPR']",
            'thresholds': "{'LQRS':'120ms', 'LPR':'200ms'}",
            'comp_op_names': "['LQRS_gt', 'LPR_gt']",
            "NORM_if_NOT": "['AVB_AIB_imp', 'RBBB_AIB_imp', 'LBBB_AIB_imp']",
            'traces': "['LPR -> AVB_AIB', 'LQRS -> RBBB_AIB', 'LQRS -> LBBB_AIB']",
            'Operations': "{'LQRS':'QRS_DUR > 120ms', 'LPR':'PR_DUR > 200ms'}",
            'Required Features': "['QRS_DUR', 'PR_DUR']",
            'diagnosis': "['AVB_AIB', 'RBBB_AIB', 'LBBB_AIB']",
            'ResultOutputs': "['AVB_AIB', 'RBBB_AIB', 'LBBB_AIB']"}

    # def rm_main(data):
    hparams = {'Imply': {'output_dims': 1, 'use_mpav': True, 'lattice_sizes': 1}}

    step2_module = EcgModule(data, {}, hparams, False)
    print(step2_module)
