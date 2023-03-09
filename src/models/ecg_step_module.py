import torch
import torch.nn as nn
from src.basic.constants import N_LEADS, SIGNAL_LEN, LPR_THRESH, LQRS_THRESH
from src.basic.rule_ml import StepModule, Imply, GE
from src.basic.dx_and_feat import Feature, get_by_str


def calc_output_shape(length_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    calculate the shape of the output from a convolutional/maxpooling layer
    """
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class EcgEmbed(StepModule):

    def __init__(self,
                 all_mid_output: dict[str, dict[str, torch.Tensor]],
                 conv_out_channels=[32, 64, 16],
                 fc_out_dim=[512, 240],
                 conv_kernel_size=5,
                 conv_stride=1,
                 pool_kernel_size=2,
                 pool_stride=2):
        super().__init__('EcgEmbed', all_mid_output)
        self.mid_output['output_dim'] = fc_out_dim[-1]
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_dim
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size

        conv_layers: list[nn.Module] = []
        fc_layers: list[nn.Module] = []

        cur_seq_len = SIGNAL_LEN  # current sequence length
        in_channels = N_LEADS

        for out_channels in conv_out_channels:
            conv_layers.append(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=conv_kernel_size,
                          stride=conv_stride))
            cur_seq_len = calc_output_shape(cur_seq_len, conv_kernel_size, conv_stride)
            conv_layers.append(nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride))
            cur_seq_len = calc_output_shape(cur_seq_len, pool_kernel_size, pool_stride)
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        input_dim = cur_seq_len * out_channels
        self.fc_input_dim = input_dim
        for output_dim in fc_out_dim:
            fc_layers.append(nn.Linear(input_dim, output_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        self.conv_layers: nn.Module = nn.Sequential(*conv_layers)
        self.fc_layers: nn.Module = nn.Sequential(*fc_layers)

    def forward(self, x):
        batched_ecg, batched_feat_vec = x
        embed = self.conv_layers(batched_ecg)
        embed = embed.view(-1, self.fc_input_dim)
        embed = self.fc_layers(embed)
        self.mid_output['embed'] = embed
        return 0, 0


class BlockModule(StepModule):

    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]]):
        super().__init__('BlockModule', all_mid_output)
        self.loss_fn = nn.BCELoss()
        self.pr_dur_ge = GE(self.mid_output, 'LPR_imp', LPR_THRESH)
        self.qrs_dur_ge = GE(self.mid_output, 'LQRS_imp', LQRS_THRESH)

        # input to imply is of shape (batch_size, 1 + feat_vec_len + embed_len)
        self.mid_output['imply_input_dim'] = 1 + len(Feature) + self.mid_output['output_dim']
        # TODO: tune AVB_imply's output_dims and lattice_sizes
        self.AVB_imply = Imply(self.mid_output,
                               consequents=['AVB'],
                               negate_consequents=[False],
                               input_dim=self.mid_output['imply_input_dim'],
                               output_dims=[128, 64],
                               use_mpa=True,
                               lattice_inc_indices=[0],
                               lattice_sizes=[4])
        self.BBB_imply = Imply(self.mid_output,
                               consequents=['LBBB', 'RBBB'],
                               negate_consequents=[False, False],
                               input_dim=self.mid_output['imply_input_dim'],
                               output_dims=[128, 64],
                               use_mpa=True)

    def forward(self, x):
        batched_ecg, batched_feat_vec = x

        PRDUR = get_by_str(batched_feat_vec, ['PRDUR'], Feature)
        QRSDUR = get_by_str(batched_feat_vec, ['QRSDUR'], Feature)

        # Apply rules such as logic and update the mid_output
        # LPR -> AVB
        self.pr_dur_ge(PRDUR)
        # LQRS -> LBBB and RBBB
        self.qrs_dur_ge(QRSDUR)

        # Assemble input for the all Imply
        common_imply_input = torch.cat((batched_feat_vec, self.all_mid_output['EcgEmbed']['embed']), dim=1)
        AVB_imply_input = torch.cat((torch.unsqueeze(self.mid_output['LPR_imp'], 0), common_imply_input), dim=1)
        BBB_imply_input = torch.cat((torch.unsqueeze(self.mid_output['LQRS_imp'], 0), common_imply_input), dim=1)

        self.AVB_imply(AVB_imply_input)
        self.BBB_imply(BBB_imply_input)

        # Compute loss
        LPR = get_by_str(batched_feat_vec, ['LPR'], Feature)
        LQRS = get_by_str(batched_feat_vec, ['LQRS'], Feature)
        objective_feat = torch.stack((LPR, LQRS), dim=1)
        feat_impressions = torch.stack((self.mid_output['LPR_imp'], self.mid_output['LQRS_imp']), dim=1)
        feat_loss = self.loss_fn(feat_impressions, objective_feat)
        self.mid_output['feat_loss'] = feat_loss  # Although normally we don't get the loss from the mid_output dict

        reg_loss = self.pr_dur_ge.reg_loss + self.qrs_dur_ge.reg_loss
        return feat_loss, reg_loss
