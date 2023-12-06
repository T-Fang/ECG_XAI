import torch
import torch.nn as nn
import pandas as pd
from src.basic.rule_ml import LT, And, Or, PipelineModule, StepModule, SeqSteps, Imply, GT, Not, ComparisonOp, get_agg_col_name
from src.basic.dx_and_feat import Diagnosis, Feature, get_by_str
from src.models.ecg_step_module import EcgStep

class AIB_Module(EcgStep):
    focused_leads: list[str] = ['[V1', 'V2', 'V5', 'V6]']
    obj_feat_names: list[str] = ['LQRS', 'LPR']
    feat_imp_names: list[str] = ['LQRS_imp', 'LPR_imp']
    comp_op_names: list[str] = ['LQRS_gt', 'LPR_gt']
    imply_names: list[str] = ['AVB_AIB_imply', 'RBBB_AIB_imply', 'LBBB_AIB_imply']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('AVB_AIB', ['AVB_AIB_imp']), ('RBBB_AIB', ['RBBB_AIB_imp']), ('LBBB_AIB', ['LBBB_AIB_imp'])]
    NORM_if_NOT: list[str] = ['AVB_AIB_imp', 'RBBB_AIB_imp', 'LBBB_AIB_imp']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        # Comparison operators
        self.LQRS_gt = GT(self, 'LQRS', 120)
        self.LPR_gt = GT(self, 'LPR', 200)

        # Imply
        self.AVB_AIB_imply = Imply(self, self.get_imply_hparams(hparams, 'LPR_imp', ['AVB_AIB_imp']))
        self.RBBB_AIB_imply = Imply(self, self.get_imply_hparams(hparams, 'LQRS_imp', ['RBBB_AIB_imp']))
        self.LBBB_AIB_imply = Imply(self, self.get_imply_hparams(hparams, 'LQRS_imp', ['LBBB_AIB_imp']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: LQRS, LPR
        QRS_DUR = get_by_str(batched_obj_feat, ['QRS_DUR'], Feature)
        PR_DUR = get_by_str(batched_obj_feat, ['PR_DUR'], Feature)

        # Comparison operators
        self.LQRS_gt(QRS_DUR)
        self.LPR_gt(PR_DUR)

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # LPR -> AVB
        self.AVB_AIB_imply(imply_input)

        # LQRS -> RBBB
        self.RBBB_AIB_imply(imply_input)

        # LQRS -> LBBB
        self.LBBB_AIB_imply(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 2: AIB Module\n')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LPR -> AVB', '', 'AVB_AIB_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LQRS -> RBBB', '', 'RBBB_AIB_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LQRS -> LBBB', '', 'LBBB_AIB_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LQRS_gt')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LPR_gt')