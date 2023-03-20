import torch
import torch.nn as nn
from src.basic.constants import N_LEADS, LEAD_TO_INDEX, ALL_LEADS, SIGNAL_LEN, LPR_THRESH, LQRS_THRESH
from src.basic.rule_ml import And, StepModule, Imply, GT, Not
from src.basic.dx_and_feat import Diagnosis, Feature, get_by_str


def calc_output_shape(length_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    calculate the shape of the output from a convolutional/maxpooling layer
    """
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class EcgStep(StepModule):

    def apply_rule(self, x) -> None:
        """
        ! Every ECG step module should implement this method !

        Apply the rule of the current step to the input batch and save the mid output to the mid_output dict
        """
        raise NotImplementedError

    def get_NORM_imp(self) -> torch.Tensor | None:
        """
        Get the NORM_imp of the current step using calculated mid_output
        """
        return None

    def compute_loss(self, x) -> torch.Tensor:
        """
        Compute the loss of the current step for the input batch and return a dict of losses
        
        e.g., ``return {'feat': 0, 'delta': 0}``
        """
        loss = {'feat': 0, 'delta': 0}
        return loss

    def save_mid_output_to_agg(self) -> torch.Tensor:
        """
        save the to-be-aggregated mid output to the mid_output dict
        """
        pass

    def forward(self, x) -> torch.Tensor:
        self.apply_rule(x)
        self.mid_output['NORM_imp'] = self.get_NORM_imp()
        loss = self.compute_loss(x)
        self.mid_output['loss'] = loss
        self.save_mid_output_to_agg()
        return loss


class BasicCNN(EcgStep):

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
        y_hat = self.fc_layers(embed)
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
        # TODO: 1dCNN + mlp (concatenate lead index before the embedding)
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

        if self.is_using_hard_rule:
            self.mid_output['embed'] = torch.zeros([batch_size, N_LEADS, self.mid_output['embed_dim_per_lead']])
            return

        lead_embeds = []
        for i in range(N_LEADS):
            lead_signal = batched_ecg[:, i, :]
            lead_embed = self.conv_layers(lead_signal)
            lead_embed = lead_embed.view(-1, self.conv_embed_dim)
            lead_index = torch.ones([batch_size, 1]) * i
            embed_with_index = torch.cat((lead_index, lead_embed), dim=1)
            lead_embed = self.fc_layers(embed_with_index)
            lead_embeds.append(lead_embed)

        embed = torch.stack(lead_embeds, dim=1)
        self.mid_output['embed'] = embed


class BlockModule(EcgStep):

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):
        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)
        self.loss_fn = nn.BCELoss()
        self.pr_dur_gt = GT(self, 'LPR_imp', LPR_THRESH)
        self.qrs_dur_gt = GT(self, 'LQRS_imp', LQRS_THRESH)
        self.NOT = Not(self)
        self.AND = And(self)

        # input to imply is of shape (batch_size, 1 + feat_vec_len + embed_len)
        self.focused_leads = hparams['focused_leads'] if 'focused_leads' in hparams else ['V1', 'V2', 'V5', 'V6']
        self.focused_leads_indices = [LEAD_TO_INDEX[lead_name] for lead_name in self.focused_leads]
        self.mid_output['imply_input_dim'] = 1 + self.all_mid_output[EcgEmbed.__name__]['embed_dim_per_lead'] * len(
            self.focused_leads)

        self.AVB_imply = Imply(self, hparams['AVB_imply'])

        #    consequents=['AVB_imp'],
        #    negate_consequents=[False],
        #    input_dim=self.mid_output['imply_input_dim'],
        #    output_dims=[128, 64],
        #    lattice_inc_indices=[0],
        #    lattice_sizes=[4]
        self.BBB_imply = Imply(self, hparams['BBB_imply'])

        #    consequents=['LBBB_imp', 'RBBB_imp'],
        #    negate_consequents=[False, False],
        #    input_dim=self.mid_output['imply_input_dim'],
        #    output_dims=[128, 64],
        #    use_mpav=True

    def apply_rule(self, x) -> None:
        batched_ecg, batched_obj_feat = x

        # Extract objective features
        PR_DUR = get_by_str(batched_obj_feat, ['PR_DUR'], Feature)
        QRS_DUR = get_by_str(batched_obj_feat, ['QRS_DUR'], Feature)

        # Apply rules such as logic and update the mid_output
        self.pr_dur_gt(PR_DUR)  # PR_DUR > 200
        self.qrs_dur_gt(QRS_DUR)  # QRS_DUR > 120

        # Assemble input for the all Imply
        embed = self.all_mid_output[EcgEmbed.__name__][
            'embed']  # embed is of size batch_size x N_LEADS x embed_dim_per_lead
        focused_embed = embed[:, self.focused_leads_indices, :]
        common_imply_input = focused_embed.flatten(start_dim=1)

        AVB_imply_input = torch.cat((torch.unsqueeze(self.mid_output['LPR_imp'], 1), common_imply_input), dim=1)
        self.AVB_imply(AVB_imply_input)  # LPR_imp -> AVB_imp

        BBB_imply_input = torch.cat((torch.unsqueeze(self.mid_output['LQRS_imp'], 1), common_imply_input), dim=1)
        self.BBB_imply(BBB_imply_input)  # LQRS_imp -> LBBB_imp ∧ RBBB_imp

    def get_NORM_imp(self) -> torch.Tensor | None:
        return self.AND([
            self.NOT(self.mid_output['AVB_imp']),
            self.NOT(self.mid_output['LBBB_imp']),
            self.NOT(self.mid_output['RBBB_imp'])
        ])

    def compute_loss(self, x) -> torch.Tensor:
        batched_ecg, batched_obj_feat = x
        LPR = get_by_str(batched_obj_feat, ['LPR'], Feature)
        LQRS = get_by_str(batched_obj_feat, ['LQRS'], Feature)
        objective_feat = torch.stack((LPR, LQRS), dim=1)
        feat_impressions = torch.stack((self.mid_output['LPR_imp'], self.mid_output['LQRS_imp']), dim=1)
        feat_loss = self.loss_fn(feat_impressions, objective_feat)
        delta_loss = self.pr_dur_gt.delta_loss + self.qrs_dur_gt.delta_loss
        loss = {'feat': feat_loss, 'delta': delta_loss}
        return loss

    def save_mid_output_to_agg(self) -> torch.Tensor:
        # save delta, w, and rho
        self.mid_output['pr_dur_gt_delta'] = self.pr_dur_gt.delta
        self.mid_output['pr_dur_gt_w'] = self.pr_dur_gt.w
        self.mid_output['qrs_dur_gt_delta'] = self.qrs_dur_gt.delta
        self.mid_output['qrs_dur_gt_w'] = self.qrs_dur_gt.w
        self.mid_output['AVB_imply_rho'] = self.AVB_imply.rho
        self.mid_output['BBB_imply_rho'] = self.BBB_imply.rho

    @property
    def extra_loss_to_log(self) -> list[tuple[str, str]]:
        return [(self.module_name, 'feat'), (self.module_name, 'delta')]

    @property
    def extra_terms_to_log(self) -> list[tuple[str, str]]:
        # log delta, w, and rho
        return [(self.module_name, 'pr_dur_gt_delta'), (self.module_name, 'pr_dur_gt_w'),
                (self.module_name, 'qrs_dur_gt_delta'), (self.module_name, 'qrs_dur_gt_w'),
                (self.module_name, 'AVB_imply_rho'), (self.module_name, 'BBB_imply_rho')]

    @property
    def mid_output_to_agg(self) -> list[tuple[str, str]]:
        return [(self.module_name, 'LPR_imp'), (self.module_name, 'LQRS_imp')]

    @property
    def dx_ensemble_dict(self) -> dict[str, list[tuple[str, str]]]:
        return {
            'AVB': [(self.module_name, 'AVB_imp')],
            'LBBB': [(self.module_name, 'LBBB_imp')],
            'RBBB': [(self.module_name, 'RBBB_imp')]
        }
