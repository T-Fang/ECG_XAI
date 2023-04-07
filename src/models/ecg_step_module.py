import torch
import torch.nn as nn
import pandas as pd
from src.basic.constants import AGE_OLD_THRESH, AXIS_LEADS, BLOCK_LEADS, BRAD_THRESH, DEEP_S_THRESH, DOM_R_THRESH, DOM_S_THRESH, INVT_THRESH, LP_THRESH_II, LQRS_WPW_THRESH, LVH_L1_OLD_THRESH, LVH_L1_YOUNG_THRESH, LVH_L2_FEMALE_THRESH, LVH_L2_MALE_THRESH, N_LEADS, LEAD_TO_INDEX, ALL_LEADS, P_LEADS, PEAK_P_THRESH_II, PEAK_P_THRESH_V1, PEAK_R_THRESH, POS_QRS_THRESH, Q_AMP_THRESH, Q_DUR_THRESH, RHYTHM_LEADS, SARRH_THRESH, SIGNAL_LEN, LPR_THRESH, LQRS_THRESH, SPR_THRESH, STD_LEADS, STD_THRESH, STE_THRESH, T_LEADS, TACH_THRESH, VH_LEADS  # noqa: E501
from src.basic.rule_ml import LT, And, Or, PipelineModule, StepModule, SeqSteps, Imply, GT, Not, ComparisonOp, get_agg_col_name
from src.basic.dx_and_feat import Diagnosis, Feature, get_by_str


def calc_output_shape(length_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    calculate the shape of the output from a convolutional/maxpooling layer
    """
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class EcgStep(StepModule):

    focused_leads: list[str] = ALL_LEADS
    obj_feat_names: list[str] = []
    feat_imp_names: list[str] = []
    comp_op_names: list[str] = []  # names of comparison operators in this EcgStep
    imply_names: list[str] = []  # names of imply modules in this EcgStep

    # list of tuples, where each tuple is (dx_name, list_of_contributing_dx_imp)
    # which respectively shows the name of a predicted diagnosis in this step
    # and a list of names for diagnosis impression that will contribute to the prediction of this diagnosis
    pred_dx_names: list[tuple[str, list[str]]] = []
    NORM_if_NOT: list[str] = []  # names of diagnosis impression for all relevant CVDs in this step

    @property
    def inverse_pred_dx_names(self) -> dict:
        inverse_dict = {}
        for dx, dx_imps in self.pred_dx_names:
            for dx_imp in dx_imps:
                inverse_dict[dx_imp] = dx
        return inverse_dict

    @property
    def extra_terms_to_agg(self) -> set[str]:
        """
        Names of extra terms (other than feat impressions) to be aggregated such as NORM_imp and imply antecedent and consequents
        """
        extra_terms: set[str] = {'NORM_imp'} if self.NORM_if_NOT else set()
        for imply_name in self.imply_names:
            imply: Imply = getattr(self, imply_name)
            if imply.antecedent not in self.feat_imp_names:
                extra_terms.add(imply.antecedent)
            extra_terms.update(imply.consequents)

        return extra_terms

    def __init__(self,
                 id: str,
                 all_mid_output: dict[str, dict[str, torch.Tensor]],
                 hparams: dict = {},
                 is_using_hard_rule: bool = False):
        super().__init__(id, all_mid_output, hparams, is_using_hard_rule)
        self.init_focused_leads()

        # * May be changed
        self.loss_fn = nn.BCELoss()

        # Common logic gates
        self.NOT = Not(self)
        self.AND = And(self)
        self.OR = Or(self)

    def init_focused_leads(self) -> None:
        if EcgEmbed.__name__ in self.all_mid_output and 'embed_dim_per_lead' in self.all_mid_output[EcgEmbed.__name__]:
            self.mid_output['focused_embed_dim'] = self.all_mid_output[EcgEmbed.__name__]['embed_dim_per_lead'] * len(
                self.focused_leads)
            self.mid_output['Imply_input_dim'] = 1 + self.mid_output['focused_embed_dim']

    def get_mlp_embed_layer(self, hparams: dict) -> nn.Module:
        # ! hparams['Imply'] should specify 'output_dims', and at least one of 'use_mpav' and 'lattice_sizes'
        # * May change to other embedding layer other than MLP
        layers = []
        input_dim = self.mid_output['focused_embed_dim']
        output_dims = hparams['Imply']['output_dims']
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        return nn.Sequential(*layers)

    @property
    def comp_ops(self) -> list[ComparisonOp]:
        return [getattr(self, comp_op_name) for comp_op_name in self.comp_op_names]

    @property
    def imply_modules(self) -> list[Imply]:
        return [getattr(self, imply_name) for imply_name in self.imply_names]

    @property
    def focused_leads_indices(self) -> list[int]:
        return [LEAD_TO_INDEX[lead_name] for lead_name in self.focused_leads]

    @property
    def use_lattice(self):
        return 'Imply' in self.hparams and 'lattice_sizes' in self.hparams['Imply'] and self.hparams['Imply'][
            'lattice_sizes']

    def apply_rule(self, x) -> None:
        """
        Apply the rule of the current step to the input batch and save the mid output to the mid_output dict
        """
        raise NotImplementedError

    def get_NORM_imp(self) -> torch.Tensor | None:
        """
        Get the NORM_imp of the current step using calculated mid_output.

        Usually a patient is NORM w.r.t. this step if no CVD is detected in this step.
        """
        if not self.NORM_if_NOT:
            return None
        return self.AND([self.NOT(self.mid_output[cvd_imp_name]) for cvd_imp_name in self.NORM_if_NOT])

    def compute_loss(self, x) -> torch.Tensor:
        """
        Compute the loss of the current step for the input batch and return a dict of losses

        e.g., ``return {'feat': 0, 'delta': 0}``
        """
        if not self.obj_feat_names or not self.feat_imp_names:
            return {'feat': 0.0, 'delta': 0.0}

        batched_ecg, batched_obj_feat = x

        # feat loss
        objective_feat = get_by_str(batched_obj_feat, self.obj_feat_names, Feature)
        feat_impressions = [self.mid_output[feat_imp_name] for feat_imp_name in self.feat_imp_names]
        feat_impressions = torch.stack(feat_impressions, dim=1)
        with torch.cuda.amp.autocast(enabled=False):
            feat_loss = self.loss_fn(feat_impressions.float(), objective_feat.float())

        # delta loss
        delta_loss = sum([comp_op.delta_loss for comp_op in self.comp_ops])

        loss = {'feat': feat_loss, 'delta': delta_loss}
        return loss

    def save_extra_terms_to_log(self) -> torch.Tensor:
        """
        Save extra to-be-logged terms to the mid_output dict
        """
        # save delta, w
        for comp_op_name in self.comp_op_names:
            comp_op = getattr(self, comp_op_name)
            self.mid_output[f'{comp_op_name}_delta'] = comp_op.delta
            self.mid_output[f'{comp_op_name}_w'] = comp_op.w

        # for imply_name in self.imply_names:
        #     imply_module = getattr(self, imply_name)
        #     self.mid_output[f'{imply_name}_rho'] = imply_module.rho

    def forward(self, x) -> torch.Tensor:
        self.apply_rule(x)
        self.mid_output['NORM_imp'] = self.get_NORM_imp()
        loss = self.compute_loss(x)
        self.mid_output['loss'] = loss
        self.save_extra_terms_to_log()
        return loss

    def get_imply_hparams(self,
                          hparams: dict,
                          antecedent: str,
                          consequents: list[str],
                          negate_atcd: bool = False) -> dict:
        negate_consequents = [False] * len(consequents)
        imply_hparams = {
            **hparams['Imply'],
            **{
                'antecedent': antecedent,
                'negate_atcd': negate_atcd,
                'consequents': consequents,
                'negate_consequents': negate_consequents,
                'input_dim': self.mid_output['Imply_input_dim']
            }
        }
        return imply_hparams

    def get_focused_embed(self) -> torch.Tensor:
        """
        Get the embeddings for the focused leads and concatenate.

        The resulting tensor will be part of the input for the all ``Imply`` modules.
        """
        embed = self.all_mid_output[EcgEmbed.__name__][
            'embed']  # embed is of size batch_size x N_LEADS x embed_dim_per_lead
        focused_embed = embed[:, self.focused_leads_indices, :]
        focused_embed = focused_embed.flatten(start_dim=1)
        return focused_embed

    @property
    def extra_loss_to_log(self) -> list[tuple[str, str]]:
        return [(self.module_name, 'feat'), (self.module_name, 'delta')]

    @property
    def extra_terms_to_log(self) -> list[tuple[str, str]]:
        # log delta, w
        return [(self.module_name, f'{comp_op_name}_delta') for comp_op_name in self.comp_op_names] + \
               [(self.module_name, f'{comp_op_name}_w') for comp_op_name in self.comp_op_names]
        #    [(self.module_name, f'{imply_name}_rho') for imply_name in self.imply_names]

    @property
    def mid_output_to_agg(self) -> list[tuple[str, str]]:
        # aggregating feature impressions and extra terms such as 'NORM_imp' and Imply's antecedents
        return [(self.module_name, feat_imp_name) for feat_imp_name in self.feat_imp_names] + \
               [(self.module_name, extra_term_name) for extra_term_name in self.extra_terms_to_agg]

    @property
    def compared_agg(self) -> list[tuple[str]]:
        compared_tuple: list[tuple[str]] = []
        for feat_imp_name, obj_feat_name in zip(self.feat_imp_names, self.obj_feat_names):
            compared_tuple.append((get_agg_col_name(self.module_name, feat_imp_name), obj_feat_name))

        inverse_pred_dx_names = self.inverse_pred_dx_names
        for imply_name in self.imply_names:
            imply: Imply = getattr(self, imply_name)
            for consequent in imply.consequents:
                compared_tuple.append((
                    get_agg_col_name(self.module_name, imply.antecedent),  # compare antecedents with all consequents
                    get_agg_col_name(self.module_name, consequent)))
                if consequent in inverse_pred_dx_names:
                    compared_tuple.append((get_agg_col_name(self.module_name,
                                                            imply.antecedent), inverse_pred_dx_names[consequent]))
        return compared_tuple

    @property
    def dx_ensemble_dict(self) -> dict[str, list[tuple[str, str]]]:
        ensemble_dict = {}
        for pred_dx_name, dx_imp_list in self.pred_dx_names:
            ensemble_dict[pred_dx_name] = [(self.module_name, dx_imp_name) for dx_imp_name in dx_imp_list]
        return ensemble_dict


class BasicCnn(EcgStep):

    def __init__(self,
                 all_mid_output: dict[str, dict[str, torch.Tensor]],
                 hparams: dict,
                 is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        self.conv_out_channels = hparams['conv_out_channels'] if 'conv_out_channels' in hparams else [32, 64, 16]
        self.fc_out_dims = hparams['fc_out_dims'] if 'fc_out_dims' in hparams else [512, 240]
        self.conv_kernel_size = hparams['conv_kernel_size'] if 'conv_kernel_size' in hparams else 5
        self.conv_stride = hparams['conv_stride'] if 'conv_stride' in hparams else 1
        self.pool_kernel_size = hparams['pool_kernel_size'] if 'pool_kernel_size' in hparams else 2
        self.pool_stride = hparams['pool_stride'] if 'pool_stride' in hparams else 2

        self.mid_output['embed_dim'] = self.fc_out_dims[-1]

        conv_layers: list[nn.Module] = []
        fc_layers: list[nn.Module] = []
        cur_seq_len = SIGNAL_LEN  # current sequence length
        in_channels = N_LEADS

        for out_channels in self.conv_out_channels:
            conv_layers.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=self.conv_kernel_size,
                          stride=self.conv_stride))
            cur_seq_len = calc_output_shape(cur_seq_len, self.conv_kernel_size, self.conv_stride)
            conv_layers.append(nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))
            cur_seq_len = calc_output_shape(cur_seq_len, self.pool_kernel_size, self.pool_stride)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        input_dim = cur_seq_len * out_channels
        self.fc_input_dim = input_dim
        for output_dim in self.fc_out_dims:
            fc_layers.append(nn.Linear(input_dim, output_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        fc_layers.append(nn.Linear(input_dim, len(Diagnosis)))

        self.conv_layers: nn.Module = nn.Sequential(*conv_layers)
        self.fc_layers: nn.Module = nn.Sequential(*fc_layers)

    def apply_rule(self, x) -> None:
        if self.is_using_hard_rule:
            return

        batched_ecg, batched_obj_feat = x
        embed = self.conv_layers(batched_ecg)
        embed = embed.view(-1, self.fc_input_dim)
        y_hat = torch.sigmoid(self.fc_layers(embed))
        self.mid_output['y_hat'] = y_hat


class EcgEmbed(EcgStep):

    def __init__(self,
                 all_mid_output: dict[str, dict[str, torch.Tensor]],
                 hparams: dict,
                 is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        self.conv_out_channels = hparams['conv_out_channels'] if 'conv_out_channels' in hparams else [32, 64, 16]
        self.fc_out_dims = hparams['fc_out_dims'] if 'fc_out_dims' in hparams else [512, 240]
        self.conv_kernel_size = hparams['conv_kernel_size'] if 'conv_kernel_size' in hparams else 5
        self.conv_stride = hparams['conv_stride'] if 'conv_stride' in hparams else 1
        self.pool_kernel_size = hparams['pool_kernel_size'] if 'pool_kernel_size' in hparams else 2
        self.pool_stride = hparams['pool_stride'] if 'pool_stride' in hparams else 2

        self.mid_output['embed_dim_per_lead'] = self.fc_out_dims[-1]

        conv_layers: list[nn.Module] = []
        fc_layers: list[nn.Module] = []
        cur_seq_len = SIGNAL_LEN  # current sequence length
        in_channels = 1
        for out_channels in self.conv_out_channels:
            conv_layers.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=self.conv_kernel_size,
                          stride=self.conv_stride))
            cur_seq_len = calc_output_shape(cur_seq_len, self.conv_kernel_size, self.conv_stride)
            conv_layers.append(nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))
            cur_seq_len = calc_output_shape(cur_seq_len, self.pool_kernel_size, self.pool_stride)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        self.conv_embed_dim = cur_seq_len * out_channels
        input_dim = self.conv_embed_dim + 1  # the 1 is the lead index
        self.fc_input_dim = input_dim
        self.mid_output['fc_input_dim'] = self.fc_input_dim

        for output_dim in self.fc_out_dims:
            fc_layers.append(nn.Linear(input_dim, output_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        self.conv_layers: nn.Module = nn.Sequential(*conv_layers)
        self.fc_layers: nn.Module = nn.Sequential(*fc_layers)

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x
        batch_size = batched_ecg.shape[0]
        self.mid_output['embed_dim_per_lead'] = 1
        if self.is_using_hard_rule:
            self.mid_output['embed'] = torch.zeros([batch_size, N_LEADS, self.mid_output['embed_dim_per_lead']],
                                                   device=batched_ecg.device)
            return

        lead_embeds = []
        for i in range(N_LEADS):
            lead_signal = batched_ecg[:, [i], :]
            lead_embed = self.conv_layers(lead_signal)
            lead_embed = lead_embed.view(-1, self.conv_embed_dim)
            lead_index = torch.ones([batch_size, 1], device=lead_embed.device) * i
            embed_with_index = torch.cat((lead_index, lead_embed), dim=1)
            lead_embed = self.fc_layers(embed_with_index)
            lead_embeds.append(lead_embed)

        embed = torch.stack(lead_embeds, dim=1)
        self.mid_output['embed'] = embed


class RhythmModule(EcgStep):

    focused_leads: list[str] = RHYTHM_LEADS
    obj_feat_names: list[str] = ['BRAD', 'TACH']
    feat_imp_names: list[str] = ['BRAD_imp', 'TACH_imp']
    comp_op_names: list[str] = ['brad_lt', 'tach_gt', 'sarrh_gt']
    imply_names: list[str] = ['ARR_imply', 'SBRAD_imply', 'STACH_imply', 'SR_imply']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('SARRH', ['SARRH_imp']),
                                                  ('AFIB', ['AFIB_imp']), ('AFLT', ['AFLT_imp']),
                                                  ('SBRAD', ['SBRAD_imp']), ('STACH', ['STACH_imp']),
                                                  ('SR', ['SR_imp'])]
    NORM_if_NOT: list[str] = ['AFIB_imp', 'AFLT_imp']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        # Comparison operators
        self.brad_lt = LT(self, 'BRAD_imp', BRAD_THRESH)
        self.tach_gt = GT(self, 'TACH_imp', TACH_THRESH)
        self.sarrh_gt = GT(self, 'SARRH_imp', SARRH_THRESH)

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)

        self.ARR_imply = Imply(self, self.get_imply_hparams(hparams, 'SINUS', ['AFIB_imp', 'AFLT_imp'], True))
        self.SBRAD_imply = Imply(self, self.get_imply_hparams(hparams, 'SBRAD_imply_atcd', ['SBRAD_imp']))
        self.STACH_imply = Imply(self, self.get_imply_hparams(hparams, 'STACH_imply_atcd', ['STACH_imp']))
        self.SR_imply = Imply(self, self.get_imply_hparams(hparams, 'SR_imply_atcd', ['SR_imp']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: HR, SINUS, RR_DIFF
        HR = get_by_str(batched_obj_feat, ['HR'], Feature)
        SINUS = get_by_str(batched_obj_feat, ['SINUS'], Feature)
        self.mid_output['SINUS'] = SINUS
        RR_DIFF = get_by_str(batched_obj_feat, ['RR_DIFF'], Feature)

        # Comparison operators
        self.brad_lt(HR)  # HR < 60 bpm
        self.tach_gt(HR)  # HR > 100 bpm
        self.sarrh_gt(RR_DIFF)  # RR_DIFF > 120 ms

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # ~SINUS -> AFIB_imp ∧ AFLT_imp
        self.ARR_imply(imply_input)

        # SINUS ∧ ~SARRH_imp ∧ BRAD_imp -> SBRAD_imp
        self.mid_output['SBRAD_imply_atcd'] = self.AND(
            [self.mid_output['SINUS'],
             self.NOT(self.mid_output['SARRH_imp']), self.mid_output['BRAD_imp']])
        self.SBRAD_imply(imply_input)

        # SINUS ∧ ~SARRH_imp ∧ TACH_imp -> STACH_imp
        self.mid_output['STACH_imply_atcd'] = self.AND(
            [self.mid_output['SINUS'],
             self.NOT(self.mid_output['SARRH_imp']), self.mid_output['TACH_imp']])
        self.STACH_imply(imply_input)

        # SINUS ∧ ~SARRH_imp ∧ ~SBRAD_imp ∧ ~STACH_imp -> SR_imp
        self.mid_output['SR_imply_atcd'] = self.AND([
            self.mid_output['SINUS'],
            self.NOT(self.mid_output['SARRH_imp']),
            self.NOT(self.mid_output['SBRAD_imp']),
            self.NOT(self.mid_output['STACH_imp'])
        ])
        self.SR_imply(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 1: Rhythm Module\n')
        self.add_obj_feat_exp(mid_output_agg, report_file_obj, 'SINUS')
        self.add_imply_exp(mid_output_agg, report_file_obj, '~SINUS -> AFIB', '', 'AFIB_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, '~SINUS -> AFLT', '', 'AFLT_imp')
        # self.add_comp_exp(mid_output_agg, report_file_obj, 'ARRH_imp')
        self.add_obj_feat_exp(mid_output_agg, report_file_obj, 'HR')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'BRAD_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'TACH_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'SINUS ∧ ~ARRH ∧ BRAD -> SBRAD', 'SBRAD_imply_atcd',
                           'SBRAD_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'SINUS ∧ ~ARRH ∧ TACH -> STACH', 'STACH_imply_atcd',
                           'STACH_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'SINUS ∧ ~ARRH ∧ ~SBRAD ∧ ~STACH -> SR', 'SR_imply_atcd',
                           'SR_imp')


class BlockModule(EcgStep):

    focused_leads: list[str] = BLOCK_LEADS
    obj_feat_names: list[str] = ['LPR', 'LQRS']
    feat_imp_names: list[str] = ['LPR_imp', 'LQRS_imp']
    comp_op_names: list[str] = ['lpr_gt', 'lqrs_gt']
    imply_names: list[str] = ['AVB_imply', 'BBB_imply_Block']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('AVB', ['AVB_imp']),
                                                  ('LBBB', ['LBBB_imp_Block']), ('RBBB', ['RBBB_imp_Block'])]
    NORM_if_NOT: list[str] = ['AVB_imp', 'LBBB_imp_Block', 'RBBB_imp_Block']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        # Comparison operators
        self.lpr_gt = GT(self, 'LPR_imp', LPR_THRESH)
        self.lqrs_gt = GT(self, 'LQRS_imp', LQRS_THRESH)

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)
        self.AVB_imply = Imply(self, self.get_imply_hparams(hparams, 'LPR_imp', ['AVB_imp']))
        self.BBB_imply_Block = Imply(self,
                                     self.get_imply_hparams(hparams, 'LQRS_imp', ['LBBB_imp_Block', 'RBBB_imp_Block']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: PR_DUR, QRS_DUR
        PR_DUR = get_by_str(batched_obj_feat, ['PR_DUR'], Feature)
        QRS_DUR = get_by_str(batched_obj_feat, ['QRS_DUR'], Feature)

        # Comparison operators
        self.lpr_gt(PR_DUR)  # PR_DUR > 200
        self.lqrs_gt(QRS_DUR)  # QRS_DUR > 120

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # LPR_imp -> AVB_imp
        self.AVB_imply(imply_input)

        # LQRS_imp -> LBBB_imp_Block ∧ RBBB_imp_Block
        self.BBB_imply_Block(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 2: Block Module\n')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LQRS_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LQRS -> LBBB', '', 'LBBB_imp_Block')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LQRS -> RBBB', '', 'RBBB_imp_Block')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LPR_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LPR -> AVB', '', 'AVB_imp')


class WPWModule(EcgStep):

    focused_leads: list[str] = ALL_LEADS
    obj_feat_names: list[str] = ['LQRS_WPW', 'SPR']
    feat_imp_names: list[str] = ['LQRS_WPW_imp', 'SPR_imp']
    comp_op_names: list[str] = ['lqrs_wpw_gt', 'spr_lt']
    imply_names: list[str] = ['WPW_imply', 'IVCD_imply']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('WPW', ['WPW_imp']), ('IVCD', ['IVCD_imp'])]
    NORM_if_NOT: list[str] = ['WPW_imp', 'IVCD_imp']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        # Comparison operators
        self.lqrs_wpw_gt = GT(self, 'LQRS_WPW_imp', LQRS_WPW_THRESH)
        self.spr_lt = LT(self, 'SPR_imp', SPR_THRESH)

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)
        self.WPW_imply = Imply(self, self.get_imply_hparams(hparams, 'WPW_imply_atcd', ['WPW_imp']))
        self.IVCD_imply = Imply(self, self.get_imply_hparams(hparams, 'IVCD_imply_atcd', ['IVCD_imp']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: PR_DUR, QRS_DUR
        PR_DUR = get_by_str(batched_obj_feat, ['PR_DUR'], Feature)
        QRS_DUR = get_by_str(batched_obj_feat, ['QRS_DUR'], Feature)

        # Comparison operators
        self.lqrs_wpw_gt(QRS_DUR)  # QRS_DUR > 110
        self.spr_lt(PR_DUR)  # PR_DUR < 120

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # LQRS_WPW_imp ∧ ~LBBB_imp_Block ∧ ~RBBB_imp_Block ∧ SPR_imp -> WPW_imp
        self.mid_output['WPW_imply_atcd'] = self.AND([
            self.mid_output['LQRS_WPW_imp'],
            self.NOT(self.all_mid_output[BlockModule.__name__]['LBBB_imp_Block']),
            self.NOT(self.all_mid_output[BlockModule.__name__]['RBBB_imp_Block']), self.mid_output['SPR_imp']
        ])
        self.WPW_imply(imply_input)

        # LQRS_WPW_imp ∧ ~LBBB_imp_Block ∧ ~RBBB_imp_Block ∧ ~SPR_imp -> IVCD_imp
        self.mid_output['IVCD_imply_atcd'] = self.AND([
            self.mid_output['LQRS_WPW_imp'],
            self.NOT(self.all_mid_output[BlockModule.__name__]['LBBB_imp_Block']),
            self.NOT(self.all_mid_output[BlockModule.__name__]['RBBB_imp_Block']),
            self.NOT(self.mid_output['SPR_imp'])
        ])
        self.IVCD_imply(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 3: WPW Module\n')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LQRS_WPW_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'SPR_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LQRS_WPW ∧ ~LBBB ∧ ~RBBB ∧ SPR -> WPW', 'WPW_imply_atcd',
                           'WPW_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LQRS_WPW ∧ ~LBBB ∧ ~RBBB ∧ ~SPR -> IVCD',
                           'IVCD_imply_atcd', 'IVCD_imp')


class STModule(EcgStep):
    focused_leads: list[str] = ALL_LEADS
    obj_feat_names: list[str] = [f'STE_{lead}' for lead in ALL_LEADS] + [f'STD_{lead}' for lead in STD_LEADS]
    feat_imp_names: list[str] = [f'STE_{lead}_imp' for lead in ALL_LEADS] + [f'STD_{lead}_imp' for lead in STD_LEADS]
    comp_op_names: list[str] = [f'ste_{lead}_gt' for lead in ALL_LEADS] + [f'std_{lead}_lt' for lead in STD_LEADS]
    imply_names: list[str] = [
        'IMI_imply_STE', 'IMI_imply_STD', 'AMI_imply_STE', 'AMI_LMI_imply_STD', 'LMI_imply_STE', 'LVH_imply_STD',
        'RVH_imply_STD'
    ]
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('IMI', ['IMI_imp_STE', 'IMI_imp_STD']),
                                                  ('AMI', ['AMI_imp_STE', 'AMI_imp_STD']),
                                                  ('LMI', ['LMI_imp_STE', 'LMI_imp_STD']), ('LVH', ['LVH_imp_STD']),
                                                  ('RVH', ['RVH_imp_STD'])]
    NORM_if_NOT: list[str] = [
        'IMI_imp_STE', 'IMI_imp_STD', 'AMI_imp_STE', 'AMI_imp_STD', 'LMI_imp_STE', 'LMI_imp_STD', 'LVH_imp_STD',
        'RVH_imp_STD'
    ]

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        self.GOR = Or(self, 2)
        # Comparison operators
        for lead in ALL_LEADS:
            self.add_module(f'ste_{lead}_gt', GT(self, f'STE_{lead}_imp', STE_THRESH))
        for lead in STD_LEADS:
            self.add_module(f'std_{lead}_lt', LT(self, f'STD_{lead}_imp', STD_THRESH))

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)
        self.IMI_imply_STE = Imply(self, self.get_imply_hparams(hparams, 'IMI_imply_STE_atcd', ['IMI_imp_STE']))
        self.AMI_imply_STE = Imply(self, self.get_imply_hparams(hparams, 'AMI_imply_STE_atcd', ['AMI_imp_STE']))
        self.LMI_imply_STE = Imply(self, self.get_imply_hparams(hparams, 'LMI_imply_STE_atcd', ['LMI_imp_STE']))

        # * Ancillary criteria
        self.IMI_imply_STD = Imply(self, self.get_imply_hparams(hparams, 'STD_aVL_imp', ['IMI_imp_STD']))
        self.AMI_LMI_imply_STD = Imply(
            self, self.get_imply_hparams(hparams, 'AMI_LMI_imply_STD_atcd', ['AMI_imp_STD', 'LMI_imp_STD']))

        self.LVH_imply_STD = Imply(self, self.get_imply_hparams(hparams, 'LVH_imply_STD_atcd', ['LVH_imp_STD']))
        self.RVH_imply_STD = Imply(self, self.get_imply_hparams(hparams, 'RVH_imply_STD_atcd', ['RVH_imp_STD']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: ST_AMP
        ST_AMP = get_by_str(batched_obj_feat, [f'ST_AMP_{lead}' for lead in ALL_LEADS],
                            Feature)  # shape: (batch_size, N_LEADS)

        # Comparison operators
        for i, lead in enumerate(ALL_LEADS):
            getattr(self, f'ste_{lead}_gt')(ST_AMP[:, i])  # STE_AMP > 0.1 mV

        for i, lead in enumerate(STD_LEADS):
            getattr(self, f'std_{lead}_lt')(ST_AMP[:, i])

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # GOR_2(STE_II_imp, STE_III_imp, STE_aVF_imp) -> IMI_imp_STE
        self.mid_output['IMI_imply_STE_atcd'] = self.GOR(
            [self.mid_output['STE_II_imp'], self.mid_output['STE_III_imp'], self.mid_output['STE_aVF_imp']])
        self.IMI_imply_STE(imply_input)

        # (STE_V1_imp ∧ STE_V2_imp) ∨ (STE_V2_imp ∧ STE_V3_imp) ∨ ... ∨ (STE_V5_imp ∧ STE_V6_imp) -> AMI_imp_STE
        self.mid_output['AMI_imply_STE_atcd'] = self.OR([
            self.AND([self.mid_output['STE_V1_imp'], self.mid_output['STE_V2_imp']]),
            self.AND([self.mid_output['STE_V2_imp'], self.mid_output['STE_V3_imp']]),
            self.AND([self.mid_output['STE_V3_imp'], self.mid_output['STE_V4_imp']]),
            self.AND([self.mid_output['STE_V4_imp'], self.mid_output['STE_V5_imp']]),
            self.AND([self.mid_output['STE_V5_imp'], self.mid_output['STE_V6_imp']])
        ])
        self.AMI_imply_STE(imply_input)

        # GOR_2(STE_I_imp, STE_aVL_imp, STE_V5_imp, STE_V6_imp) -> LMI_imp_STE
        self.mid_output['LMI_imply_STE_atcd'] = self.GOR([
            self.mid_output['STE_I_imp'], self.mid_output['STE_aVL_imp'], self.mid_output['STE_V5_imp'],
            self.mid_output['STE_V6_imp']
        ])
        self.LMI_imply_STE(imply_input)

        # * Ancillary criteria
        # STD_aVL_imp -> IMI_imp_STD
        self.IMI_imply_STD(imply_input)

        # GOR_2(STD_II_imp, STD_III_imp, STD_aVF_imp) -> AMI_imp_STD ∧ LMI_imp_STD
        self.mid_output['AMI_LMI_imply_STD_atcd'] = self.GOR(
            [self.mid_output['STD_II_imp'], self.mid_output['STD_III_imp'], self.mid_output['STD_aVF_imp']])
        self.AMI_LMI_imply_STD(imply_input)

        # STD_V5_imp ∨ STD_V6_imp -> LVH_imp_STD
        self.mid_output['LVH_imply_STD_atcd'] = self.OR([self.mid_output['STD_V5_imp'], self.mid_output['STD_V6_imp']])
        self.LVH_imply_STD(imply_input)

        # STD_V1_imp ∧ STD_V2_imp ∧ STD_V3_imp -> RVH_imp_STD
        self.mid_output['RVH_imply_STD_atcd'] = self.AND(
            [self.mid_output['STD_V1_imp'], self.mid_output['STD_V2_imp'], self.mid_output['STD_V3_imp']])
        self.RVH_imply_STD(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 4: ST Module\n')
        for lead in ALL_LEADS:
            self.add_comp_exp(mid_output_agg, report_file_obj, f'STE_{lead}_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'GOR_2(STE_II, STE_III, STE_aVF) -> IMI',
                           'IMI_imply_STE_atcd', 'IMI_imp_STE')
        self.add_imply_exp(mid_output_agg, report_file_obj,
                           '(STE_V1 ∧ STE_V2) ∨ (STE_V2 ∧ STE_V3) ∨ ... ∨ (STE_V5 ∧ STE_V6) -> AMI',
                           'AMI_imply_STE_atcd', 'AMI_imp_STE')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'GOR_2(STE_I, STE_aVL, STE_V5, STE_V6) -> LMI',
                           'LMI_imply_STE_atcd', 'LMI_imp_STE')
        report_file_obj.write('### Ancillary criteria using STD\n')
        for lead in STD_LEADS:
            self.add_comp_exp(mid_output_agg, report_file_obj, f'STD_{lead}_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'STD_aVL -> IMI', '', 'IMI_imp_STD')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'GOR_2(STD_II, STD_III, STD_aVF) -> AMI',
                           'AMI_LMI_imply_STD_atcd', 'AMI_imp_STD')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'GOR_2(STD_II, STD_III, STD_aVF) -> LMI',
                           'AMI_LMI_imply_STD_atcd', 'LMI_imp_STD')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'STD_V5 ∨ STD_V6 -> LVH', 'LVH_imply_STD_atcd',
                           'LVH_imp_STD')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'STD_V1 ∧ STD_V2 ∧ STD_V3 -> RVH', 'RVH_imply_STD_atcd',
                           'RVH_imp_STD')


class QRModule(EcgStep):
    focused_leads: list[str] = ALL_LEADS
    obj_feat_names: list[str] = [f'PATH_Q_{lead}' for lead in ALL_LEADS]
    feat_imp_names: list[str] = [f'PATH_Q_{lead}_imp' for lead in ALL_LEADS]
    comp_op_names: list[str] = [f'lq_{lead}_gt' for lead in ALL_LEADS] + [f'deep_q_{lead}_lt' for lead in ALL_LEADS]
    imply_names: list[str] = ['PRWP_imply', 'IMI_imply_Q', 'AMI_imply_Q', 'LMI_imply_Q']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('LVH', ['LVH_imp_PRWP']),
                                                  ('LBBB', ['LBBB_imp_PRWP']), ('IMI', ['IMI_imp_Q']),
                                                  ('AMI', ['AMI_imp_Q', 'AMI_imp_PRWP']), ('LMI', ['LMI_imp_Q'])]
    NORM_if_NOT: list[str] = ['AMI_imp_PRWP', 'LVH_imp_PRWP', 'LBBB_imp_PRWP', 'IMI_imp_Q', 'AMI_imp_Q', 'LMI_imp_Q']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        self.GOR = Or(self, 2)
        # Comparison operators
        for lead in ALL_LEADS:
            self.add_module(f'lq_{lead}_gt', GT(self, f'LQ_{lead}_imp', Q_DUR_THRESH))
        for lead in ALL_LEADS:
            self.add_module(f'deep_q_{lead}_lt', LT(self, f'DEEP_Q_{lead}_imp', Q_AMP_THRESH[lead]))

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)

        # * Ancillary criteria
        self.PRWP_imply = Imply(
            self, self.get_imply_hparams(hparams, 'PRWP', ['AMI_imp_PRWP', 'LVH_imp_PRWP', 'LBBB_imp_PRWP']))
        self.IMI_imply_Q = Imply(self, self.get_imply_hparams(hparams, 'IMI_imply_Q_atcd', ['IMI_imp_Q']))
        self.AMI_imply_Q = Imply(self, self.get_imply_hparams(hparams, 'AMI_imply_Q_atcd', ['AMI_imp_Q']))
        self.LMI_imply_Q = Imply(self, self.get_imply_hparams(hparams, 'LMI_imply_Q_atcd', ['LMI_imp_Q']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: Q_DUR, Q_AMP, and PRWP

        Q_DUR = get_by_str(batched_obj_feat, [f'Q_DUR_{lead}' for lead in ALL_LEADS], Feature)
        # shape: (batch_size, N_LEADS)
        Q_AMP = get_by_str(batched_obj_feat, [f'Q_AMP_{lead}' for lead in ALL_LEADS], Feature)
        PRWP = get_by_str(batched_obj_feat, ['PRWP'], Feature)
        self.mid_output['PRWP'] = PRWP

        # Comparison operators
        for i, lead in enumerate(ALL_LEADS):
            getattr(self, f'lq_{lead}_gt')(Q_DUR[:, i])
            getattr(self, f'deep_q_{lead}_lt')(Q_AMP[:, i])

        # Get PATH_Q_{lead}_imp for all leads: PATH_Q_{lead}_imp = LQ_{lead}_imp ∨ DEEP_Q_{lead}_imp
        for lead in ALL_LEADS:
            self.mid_output[f'PATH_Q_{lead}_imp'] = self.OR(
                [self.mid_output[f'LQ_{lead}_imp'], self.mid_output[f'DEEP_Q_{lead}_imp']])

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # PRWP -> AMI_imp_PRWP ∧ LVH_imp_PRWP ∧ LBBB_imp_PRWP
        self.PRWP_imply(imply_input)

        # GOR_2(PATH_Q_II_imp, PATH_Q_III_imp, PATH_Q_aVF_imp) -> IMI_imp_Q
        self.mid_output['IMI_imply_Q_atcd'] = self.GOR(
            [self.mid_output['PATH_Q_II_imp'], self.mid_output['PATH_Q_III_imp'], self.mid_output['PATH_Q_aVF_imp']])
        self.IMI_imply_Q(imply_input)

        # PATH_Q_V1_imp ∧ PATH_Q_V2_imp ∧ PATH_Q_V3_imp ∧ PATH_Q_V4_imp -> AMI_imp_Q
        self.mid_output['AMI_imply_Q_atcd'] = self.AND([
            self.mid_output['PATH_Q_V1_imp'], self.mid_output['PATH_Q_V2_imp'], self.mid_output['PATH_Q_V3_imp'],
            self.mid_output['PATH_Q_V4_imp']
        ])
        self.AMI_imply_Q(imply_input)

        # GOR_2(PATH_Q_I_imp, PATH_Q_aVL_imp, PATH_Q_V5_imp, PATH_Q_V6_imp) -> LMI_imp_Q
        self.mid_output['LMI_imply_Q_atcd'] = self.GOR([
            self.mid_output['PATH_Q_I_imp'], self.mid_output['PATH_Q_aVL_imp'], self.mid_output['PATH_Q_V5_imp'],
            self.mid_output['PATH_Q_V6_imp']
        ])
        self.LMI_imply_Q(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 5: QR Module\n')
        report_file_obj.write('### Ancillary criteria using Pathological Q wave and Poor R wave Progression\n')

        self.add_obj_feat_exp(mid_output_agg, report_file_obj, 'PRWP')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'PRWP -> AMI', '', 'AMI_imp_PRWP')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'PRWP -> LVH', '', 'LVH_imp_PRWP')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'PRWP -> LBBB', '', 'LBBB_imp_PRWP')

        for lead in ALL_LEADS:
            self.add_comp_exp(mid_output_agg, report_file_obj, f'PATH_Q_{lead}_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'GOR_2(PATH_Q_II, PATH_Q_III, PATH_Q_aVF) -> IMI',
                           'IMI_imply_Q_atcd', 'IMI_imp_Q')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'PATH_Q_V1 ∧ PATH_Q_V2 ∧ PATH_Q_V3 ∧ PATH_Q_V4 -> AMI',
                           'AMI_imply_Q_atcd', 'AMI_imp_Q')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'GOR_2(PATH_Q_I, PATH_Q_aVL, PATH_Q_V5, PATH_Q_V6) -> LMI',
                           'LMI_imply_Q_atcd', 'LMI_imp_Q')


class PModule(EcgStep):

    focused_leads: list[str] = P_LEADS
    obj_feat_names: list[str] = ['LP_II', 'PEAK_P_II', 'PEAK_P_V1']
    feat_imp_names: list[str] = ['LP_II_imp', 'PEAK_P_II_imp', 'PEAK_P_V1_imp']
    comp_op_names: list[str] = [
        'lp_II_gt',
        'peak_p_II_gt',
        'peak_p_V1_gt',
    ]
    imply_names: list[str] = ['LAE_imply', 'RAE_imply', 'LVH_imply_P', 'RVH_imply_P']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('LAE', ['LAE_imp']), ('RAE', ['RAE_imp']),
                                                  ('LVH', ['LVH_imp_P']), ('RVH', ['RVH_imp_P'])]
    NORM_if_NOT: list[str] = ['LAE_imp', 'RAE_imp', 'LVH_imp_P', 'RVH_imp_P']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        # Comparison operators
        self.lp_II_gt = GT(self, 'LP_II_imp', LP_THRESH_II)
        self.peak_p_II_gt = GT(self, 'PEAK_P_II_imp', PEAK_P_THRESH_II)
        self.peak_p_V1_gt = GT(self, 'PEAK_P_V1_imp', PEAK_P_THRESH_V1)

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)
        self.LAE_imply = Imply(self, self.get_imply_hparams(hparams, 'LP_II_imp', ['LAE_imp']))
        self.RAE_imply = Imply(self, self.get_imply_hparams(hparams, 'RAE_imply_atcd', ['RAE_imp']))

        # * Ancillary criteria
        self.LVH_imply_P = Imply(self, self.get_imply_hparams(hparams, 'LAE_imp', ['LVH_imp_P']))
        self.RVH_imply_P = Imply(self, self.get_imply_hparams(hparams, 'RAE_imp', ['RVH_imp_P']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: P_DUR_II, P_AMP_II, P_AMP_V1
        P_DUR_II = get_by_str(batched_obj_feat, ['P_DUR_II'], Feature)
        P_AMP_II = get_by_str(batched_obj_feat, ['P_AMP_II'], Feature)
        P_AMP_V1 = get_by_str(batched_obj_feat, ['P_AMP_V1'], Feature)

        # Comparison operators
        self.lp_II_gt(P_DUR_II)  # P_DUR_II > 110
        self.peak_p_II_gt(P_AMP_II)  # P_AMP_II > 0.25
        self.peak_p_V1_gt(P_AMP_V1)  # P_AMP_V1 > 0.15

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # LP_II_imp -> LAE_imp
        self.LAE_imply(imply_input)

        # PEAK_P_II_imp ∨ PEAK_P_V1_imp -> RAE_imp
        self.mid_output['RAE_imply_atcd'] = self.OR(
            [self.mid_output['PEAK_P_II_imp'], self.mid_output['PEAK_P_V1_imp']])
        self.RAE_imply(imply_input)

        # * Ancillary criteria
        # LAE_imp -> LVH_imp_P
        self.LVH_imply_P(imply_input)

        # RAE_imp -> RVH_imp_P
        self.RVH_imply_P(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 6: P Module\n')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LP_II_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LP_II -> LAE', '', 'LAE_imp')

        self.add_comp_exp(mid_output_agg, report_file_obj, 'PEAK_P_II_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'PEAK_P_V1_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'PEAK_P_II ∨ PEAK_P_V1 -> RAE', 'RAE_imply_atcd', 'RAE_imp')

        report_file_obj.write('### Ancillary criteria using LAE and RAE\n')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LAE -> LVH', '', 'LVH_imp_P')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'RAE -> RVH', '', 'RVH_imp_P')


class VHModule(EcgStep):

    focused_leads: list[str] = VH_LEADS
    obj_feat_names: list[str] = [
        'AGE_OLD', 'LVH_L1_OLD', 'LVH_L1_YOUNG', 'LVH_L2_MALE', 'LVH_L2_FEMALE', 'PEAK_R_V1', 'DEEP_S_V5', 'DEEP_S_V6',
        'DOM_R_V1', 'DOM_S_V5', 'DOM_S_V6'
    ]
    feat_imp_names: list[str] = [
        'AGE_OLD_imp', 'LVH_L1_OLD_imp', 'LVH_L1_YOUNG_imp', 'LVH_L2_MALE_imp', 'LVH_L2_FEMALE_imp', 'PEAK_R_V1_imp',
        'DEEP_S_V5_imp', 'DEEP_S_V6_imp', 'DOM_R_V1_imp', 'DOM_S_V5_imp', 'DOM_S_V6_imp'
    ]
    comp_op_names: list[str] = [
        'age_old_gt', 'lvh_l1_old_gt', 'lvh_l1_young_gt', 'lvh_l2_male_gt', 'lvh_l2_female_gt', 'peak_r_v1_gt',
        'deep_s_v5_gt', 'deep_s_v6_gt', 'dom_r_v1_gt', 'dom_s_v5_lt', 'dom_s_v6_lt'
    ]
    imply_names: list[str] = ['LVH_imply_VH', 'RVH_imply_VH']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('LVH', ['LVH_imp_VH']),
                                                  ('RVH', ['RVH_imp_VH'])]
    NORM_if_NOT: list[str] = ['LVH_imp_VH', 'RVH_imp_VH']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)
        self.GOR = Or(self, 2)

        # Comparison operators
        self.age_old_gt = GT(self, 'AGE_OLD_imp', AGE_OLD_THRESH)
        self.lvh_l1_old_gt = GT(self, 'LVH_L1_OLD_imp', LVH_L1_OLD_THRESH)
        self.lvh_l1_young_gt = GT(self, 'LVH_L1_YOUNG_imp', LVH_L1_YOUNG_THRESH)
        self.lvh_l2_male_gt = GT(self, 'LVH_L2_MALE_imp', LVH_L2_MALE_THRESH)
        self.lvh_l2_female_gt = GT(self, 'LVH_L2_FEMALE_imp', LVH_L2_FEMALE_THRESH)
        self.peak_r_v1_gt = GT(self, 'PEAK_R_V1_imp', PEAK_R_THRESH)
        self.deep_s_v5_gt = GT(self, 'DEEP_S_V5_imp', DEEP_S_THRESH)
        self.deep_s_v6_gt = GT(self, 'DEEP_S_V6_imp', DEEP_S_THRESH)
        self.dom_r_v1_gt = GT(self, 'DOM_R_V1_imp', DOM_R_THRESH)
        self.dom_s_v5_lt = LT(self, 'DOM_S_V5_imp', DOM_S_THRESH)
        self.dom_s_v6_lt = LT(self, 'DOM_S_V6_imp', DOM_S_THRESH)

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)
        self.LVH_imply_VH = Imply(self, self.get_imply_hparams(hparams, 'LVH_imply_VH_atcd', ['LVH_imp_VH']))
        self.RVH_imply_VH = Imply(self, self.get_imply_hparams(hparams, 'RVH_imply_VH_atcd', ['RVH_imp_VH']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: AGE, MALE, R_AMP_aVL, R_AMP_V1, R_AMP_V6, S_AMP_V1, S_AMP_V3, S_AMP_V5, S_AMP_V6, RS_RATIO_V1, RS_RATIO_V5, RS_RATIO_V6, RAD # noqa: E501
        AGE = get_by_str(batched_obj_feat, ['AGE'], Feature)
        MALE = get_by_str(batched_obj_feat, ['MALE'], Feature)
        R_AMP_aVL = get_by_str(batched_obj_feat, ['R_AMP_aVL'], Feature)
        R_AMP_V1 = get_by_str(batched_obj_feat, ['R_AMP_V1'], Feature)
        R_AMP_V6 = get_by_str(batched_obj_feat, ['R_AMP_V6'], Feature)
        S_AMP_V1 = get_by_str(batched_obj_feat, ['S_AMP_V1'], Feature)
        S_AMP_V3 = get_by_str(batched_obj_feat, ['S_AMP_V3'], Feature)
        S_AMP_V5 = get_by_str(batched_obj_feat, ['S_AMP_V5'], Feature)
        S_AMP_V6 = get_by_str(batched_obj_feat, ['S_AMP_V6'], Feature)
        RS_RATIO_V1 = get_by_str(batched_obj_feat, ['RS_RATIO_V1'], Feature)
        RS_RATIO_V5 = get_by_str(batched_obj_feat, ['RS_RATIO_V5'], Feature)
        RS_RATIO_V6 = get_by_str(batched_obj_feat, ['RS_RATIO_V6'], Feature)
        RAD = get_by_str(batched_obj_feat, ['RAD'], Feature)

        # Comparison operators
        self.age_old_gt(AGE)  # AGE > 30
        self.lvh_l1_old_gt(S_AMP_V1 + R_AMP_V6)  # S_AMP_V1 + R_AMP_V6 > 3.5 mV
        self.lvh_l1_young_gt(S_AMP_V1 + R_AMP_V6)  # S_AMP_V1 + R_AMP_V6 > 4 mV
        self.lvh_l2_male_gt(R_AMP_aVL + S_AMP_V3)  # R_AMP_aVL + S_AMP_V3 > 2.4 mV
        self.lvh_l2_female_gt(R_AMP_aVL + S_AMP_V3)  # R_AMP_aVL + S_AMP_V3 > 1.8 mV
        self.peak_r_v1_gt(R_AMP_V1)  # R_AMP_V1 > 0.7 mV
        self.deep_s_v5_gt(S_AMP_V5)  # S_AMP_V5 > 0.7 mV
        self.deep_s_v6_gt(S_AMP_V6)  # S_AMP_V6 > 0.7 mV
        self.dom_r_v1_gt(RS_RATIO_V1)  # RS_RATIO_V1 > 1
        self.dom_s_v5_lt(RS_RATIO_V5)  # RS_RATIO_V5 < 1
        self.dom_s_v6_lt(RS_RATIO_V6)  # RS_RATIO_V6 < 1

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # (AGE_OLD_imp ∧ LVH_L1_OLD_imp) ∨ (~AGE_OLD_imp ∧ LVH_L1_YOUNG_imp) ∨ (MALE ∧ LVH_L2_MALE_imp) ∨ (~MALE ∧ LVH_L2_FEMALE_imp) -> LVH_imp_VH
        self.mid_output['LVH_imply_VH_atcd'] = self.OR([
            self.AND([self.mid_output['AGE_OLD_imp'], self.mid_output['LVH_L1_OLD_imp']]),
            self.AND([self.NOT(self.mid_output['AGE_OLD_imp']), self.mid_output['LVH_L1_YOUNG_imp']]),
            self.AND([MALE, self.mid_output['LVH_L2_MALE_imp']]),
            self.AND([self.NOT(MALE), self.mid_output['LVH_L2_FEMALE_imp']]),
        ])
        self.LVH_imply_VH(imply_input)

        # GOR_2(PEAK_R_V1_imp, DEEP_S_V5_imp ∨ DEEP_S_V6_imp, DOM_R_V1_imp, DOM_S_V5_imp ∨ DOM_S_V6_imp, RAD) -> RVH_imp_VH
        self.mid_output['RVH_imply_VH_atcd'] = self.GOR([
            self.mid_output['PEAK_R_V1_imp'],
            self.OR([self.mid_output['DEEP_S_V5_imp'], self.mid_output['DEEP_S_V6_imp']]),
            self.mid_output['DOM_R_V1_imp'],
            self.OR([self.mid_output['DOM_S_V5_imp'], self.mid_output['DOM_S_V6_imp']]),
            RAD,
        ])
        self.RVH_imply_VH(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 7: VH Module\n')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'AGE_OLD_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LVH_L1_OLD_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LVH_L1_YOUNG_imp')
        self.add_obj_feat_exp(mid_output_agg, report_file_obj, 'MALE')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LVH_L2_MALE_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'LVH_L2_FEMALE_imp')
        self.add_imply_exp(
            mid_output_agg, report_file_obj,
            '(AGE_OLD ∧ LVH_L1_OLD) ∨ (~AGE_OLD ∧ LVH_L1_YOUNG) ∨ (MALE ∧ LVH_L2_MALE) ∨ (~MALE ∧ LVH_L2_FEMALE) -> LVH',
            'LVH_imply_VH_atcd', 'LVH_imp_VH')

        self.add_comp_exp(mid_output_agg, report_file_obj, 'PEAK_R_V1_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'DEEP_S_V5_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'DEEP_S_V6_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'DOM_R_V1_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'DOM_S_V5_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'DOM_S_V6_imp')
        self.add_obj_feat_exp(mid_output_agg, report_file_obj, 'RAD')
        self.add_imply_exp(mid_output_agg, report_file_obj,
                           'GOR_2(PEAK_R_V1, DEEP_S_V5 ∨ DEEP_S_V6, DOM_R_V1, DOM_S_V5 ∨ DOM_S_V6, RAD) -> RVH',
                           'RVH_imply_VH_atcd', 'RVH_imp_VH')


class TModule(EcgStep):
    focused_leads: list[str] = T_LEADS
    obj_feat_names: list[str] = [f'INVT_{lead}' for lead in T_LEADS]
    feat_imp_names: list[str] = [f'INVT_{lead}_imp' for lead in T_LEADS]
    comp_op_names: list[str] = [f'invt_{lead}_lt' for lead in T_LEADS]
    imply_names: list[str] = ['MI_imply_T', 'LVH_imply_T', 'RVH_imply_T']
    pred_dx_names: list[tuple[str,
                              list[str]]] = [('NORM', ['NORM_imp']), ('IMI', ['IMI_imp_T']), ('AMI', ['AMI_imp_T']),
                                             ('LMI', ['LMI_imp_T']), ('LVH', ['LVH_imp_T']), ('RVH', ['RVH_imp_T'])]

    NORM_if_NOT: list[str] = ['IMI_imp_T', 'AMI_imp_T', 'LMI_imp_T', 'LVH_imp_T', 'RVH_imp_T']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)
        self.GOR = Or(self, 2)

        # Comparison operators
        for lead in T_LEADS:
            self.add_module(f'invt_{lead}_lt', LT(self, f'INVT_{lead}_imp', INVT_THRESH))

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)
        self.MI_imply_T = Imply(
            self, self.get_imply_hparams(hparams, 'MI_imply_T_atcd', ['IMI_imp_T', 'AMI_imp_T', 'LMI_imp_T']))

        # * Ancillary criteria
        self.LVH_imply_T = Imply(self, self.get_imply_hparams(hparams, 'LVH_imply_T_atcd', ['LVH_imp_T']))
        self.RVH_imply_T = Imply(self, self.get_imply_hparams(hparams, 'RVH_imply_T_atcd', ['RVH_imp_T']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: ST_AMP, T_AMP
        T_AMP = get_by_str(batched_obj_feat, [f'T_AMP_{lead}' for lead in T_LEADS],
                           Feature)  # shape: (batch_size, len(T_LEADS))

        # Comparison operators
        for i, lead in enumerate(T_LEADS):
            getattr(self, f'invt_{lead}_lt')(T_AMP[:, i])  # T_AMP < 0 mV

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # GOR_2([INVT_imp_x ∧ (STE_x_imp ∨ STD_x_imp), for x ∈ {I, II, V3-V6}]) -> IMI_imp_T ∧ AMI_imp_T ∧ LMI_imp_T
        self.mid_output['MI_imply_T_atcd'] = self.GOR([
            self.AND([
                self.mid_output['INVT_I_imp'],
                self.OR([
                    self.all_mid_output[STModule.__name__]['STE_I_imp'],
                    self.all_mid_output[STModule.__name__]['STD_I_imp']
                ])
            ]),
            self.AND([
                self.mid_output['INVT_II_imp'],
                self.OR([
                    self.all_mid_output[STModule.__name__]['STE_II_imp'],
                    self.all_mid_output[STModule.__name__]['STD_II_imp']
                ])
            ]),
            self.AND([
                self.mid_output['INVT_V3_imp'],
                self.OR([
                    self.all_mid_output[STModule.__name__]['STE_V3_imp'],
                    self.all_mid_output[STModule.__name__]['STD_V3_imp']
                ])
            ]),
            self.AND([
                self.mid_output['INVT_V4_imp'],
                self.OR([
                    self.all_mid_output[STModule.__name__]['STE_V4_imp'],
                    self.all_mid_output[STModule.__name__]['STD_V4_imp']
                ])
            ]),
            self.AND([
                self.mid_output['INVT_V5_imp'],
                self.OR([
                    self.all_mid_output[STModule.__name__]['STE_V5_imp'],
                    self.all_mid_output[STModule.__name__]['STD_V5_imp']
                ])
            ]),
            self.AND([
                self.mid_output['INVT_V6_imp'],
                self.OR([
                    self.all_mid_output[STModule.__name__]['STE_V6_imp'],
                    self.all_mid_output[STModule.__name__]['STD_V6_imp']
                ])
            ])
        ])
        self.MI_imply_T(imply_input)

        # * Ancillary criteria
        # INVT_V5_imp ∨ INVT_V6_imp -> LVH_imp_T
        self.mid_output['LVH_imply_T_atcd'] = self.OR([self.mid_output['INVT_V5_imp'], self.mid_output['INVT_V6_imp']])
        self.LVH_imply_T(imply_input)

        # INVT_V1_imp ∧ INVT_V2_imp ∧ INVT_V3_imp -> RVH_imp_T
        self.mid_output['RVH_imply_T_atcd'] = self.AND(
            [self.mid_output['INVT_V1_imp'], self.mid_output['INVT_V2_imp'], self.mid_output['INVT_V3_imp']])
        self.RVH_imply_T(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 8: T Module\n')
        for lead in T_LEADS:
            self.add_comp_exp(mid_output_agg, report_file_obj, f'INVT_{lead}_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj,
                           'GOR_2([INVT_x ∧ (STE_x ∨ STD_x), for x ∈ {I, II, V3-V6}]) -> IMI',
                           'MI_imply_T_atcd', 'IMI_imp_T')
        self.add_imply_exp(mid_output_agg, report_file_obj,
                           'GOR_2([INVT_x ∧ (STE_x ∨ STD_x), for x ∈ {I, II, V3-V6}]) -> AMI', 'MI_imply_T_atcd',
                           'AMI_imp_T')
        self.add_imply_exp(mid_output_agg, report_file_obj,
                           'GOR_2([INVT_x ∧ (STE_x ∨ STD_x), for x ∈ {I, II, V3-V6}]) -> LMI', 'MI_imply_T_atcd',
                           'LMI_imp_T')

        report_file_obj.write('### Ancillary criteria using inverted T wave\n')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'INVT_V5 ∨ INVT_V6 -> LVH', 'LVH_imply_T_atcd', 'LVH_imp_T')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'INVT_V1 ∧ INVT_V2 ∧ INVT_V3 -> RVH', 'RVH_imply_T_atcd',
                           'RVH_imp_T')


class AxisModule(EcgStep):
    focused_leads: list[str] = AXIS_LEADS
    obj_feat_names: list[str] = ['POS_QRS_I', 'POS_QRS_aVF', 'NORM_AXIS', 'LAD', 'RAD']
    feat_imp_names: list[str] = ['POS_QRS_I_imp', 'POS_QRS_aVF_imp', 'NORM_AXIS_imp', 'LAD_imp', 'RAD_imp']
    comp_op_names: list[str] = ['pos_qrs_I_gt', 'pos_qrs_aVF_gt']
    imply_names: list[str] = ['LAFB_imply', 'LPFB_imply']
    pred_dx_names: list[tuple[str, list[str]]] = [('NORM', ['NORM_imp']), ('LAFB', ['LAFB_imp']),
                                                  ('LPFB', ['LPFB_imp'])]
    NORM_if_NOT: list[str] = ['LAFB_imp', 'LPFB_imp']

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)

        # Comparison operators
        self.pos_qrs_I_gt = GT(self, 'POS_QRS_I_imp', POS_QRS_THRESH)
        self.pos_qrs_aVF_gt = GT(self, 'POS_QRS_aVF_imp', POS_QRS_THRESH)

        # Imply
        if not self.use_lattice:
            self.imply_decision_embed_layer = self.get_mlp_embed_layer(hparams)

        self.LAFB_imply = Imply(self, self.get_imply_hparams(hparams, 'LAD_imp', ['LAFB_imp']))
        self.LPFB_imply = Imply(self, self.get_imply_hparams(hparams, 'RAD_imp', ['LPFB_imp']))

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features: QRS_SUM_I, QRS_SUM_aVF
        QRS_SUM_I = get_by_str(batched_obj_feat, ['QRS_SUM_I'], Feature)
        QRS_SUM_aVF = get_by_str(batched_obj_feat, ['QRS_SUM_aVF'], Feature)

        # Comparison operators
        self.pos_qrs_I_gt(QRS_SUM_I)
        self.pos_qrs_aVF_gt(QRS_SUM_aVF)

        # NORM_AXIS_imp := POS_QRS_I ∧ POS_QRS_aVF
        self.mid_output['NORM_AXIS_imp'] = self.AND(
            [self.mid_output['POS_QRS_I_imp'], self.mid_output['POS_QRS_aVF_imp']])
        # LAD_imp := POS_QRS_I ∧ ~POS_QRS_aVF
        self.mid_output['LAD_imp'] = self.AND(
            [self.mid_output['POS_QRS_I_imp'],
             self.NOT(self.mid_output['POS_QRS_aVF_imp'])])
        # RAD_imp := ~POS_QRS_I ∧ POS_QRS_aVF
        self.mid_output['RAD_imp'] = self.AND(
            [self.NOT(self.mid_output['POS_QRS_I_imp']), self.mid_output['POS_QRS_aVF_imp']])

        # Imply
        focused_embed = self.get_focused_embed()
        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)
        imply_input = (focused_embed, decision_embed)

        # LAD_imp -> LAFB_imp
        self.LAFB_imply(imply_input)

        # RAD_imp -> LPFB_imp
        self.LPFB_imply(imply_input)

    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):
        report_file_obj.write('## Step 9: Axis Module\n')
        report_file_obj.write('### Ancillary criteria using electrical axis\n')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'POS_QRS_I_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'POS_QRS_aVF_imp')
        self.add_comp_exp(mid_output_agg, report_file_obj, 'NORM_AXIS_imp')

        self.add_comp_exp(mid_output_agg, report_file_obj, 'LAD_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'LAD -> LAFB', '', 'LAFB_imp')

        self.add_comp_exp(mid_output_agg, report_file_obj, 'RAD_imp')
        self.add_imply_exp(mid_output_agg, report_file_obj, 'RAD -> LPFB', '', 'LPFB_imp')


class BasicCnnPipeline(PipelineModule):

    def _build_pipeline(self):
        basic_cnn = BasicCnn(self.all_mid_output, self.hparams.hparams[BasicCnn.__name__], self.is_using_hard_rule)
        steps = [basic_cnn]
        pipeline = SeqSteps(SeqSteps.__name__, self.all_mid_output, steps, {}, self.is_using_hard_rule)
        return pipeline

    def forward(self, batched_ecg: torch.Tensor, batched_obj_feat: torch.Tensor):
        self.pipeline((batched_ecg, batched_obj_feat))
        return self.all_mid_output[BasicCnn.__name__]['y_hat']


class EcgPipeline(PipelineModule):

    def _build_pipeline(self):
        ecg_embed = EcgEmbed(self.all_mid_output, self.hparams.hparams[EcgEmbed.__name__], self.is_using_hard_rule)
        rhythm = RhythmModule(self.all_mid_output, self.hparams.hparams[RhythmModule.__name__], self.is_using_hard_rule)
        block = BlockModule(self.all_mid_output, self.hparams.hparams[BlockModule.__name__], self.is_using_hard_rule)
        wpw = WPWModule(self.all_mid_output, self.hparams.hparams[WPWModule.__name__], self.is_using_hard_rule)
        st = STModule(self.all_mid_output, self.hparams.hparams[STModule.__name__], self.is_using_hard_rule)
        qr = QRModule(self.all_mid_output, self.hparams.hparams[QRModule.__name__], self.is_using_hard_rule)
        p = PModule(self.all_mid_output, self.hparams.hparams[PModule.__name__], self.is_using_hard_rule)
        vh = VHModule(self.all_mid_output, self.hparams.hparams[VHModule.__name__], self.is_using_hard_rule)
        t = TModule(self.all_mid_output, self.hparams.hparams[TModule.__name__], self.is_using_hard_rule)
        axis = AxisModule(self.all_mid_output, self.hparams.hparams[AxisModule.__name__], self.is_using_hard_rule)

        steps = [ecg_embed, rhythm, block, wpw, st, qr, p, vh, t, axis]
        pipeline = SeqSteps(SeqSteps.__name__, self.all_mid_output, steps, {}, self.is_using_hard_rule)
        return pipeline
