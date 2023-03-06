import torch.nn as nn
from src.basic.constants import N_LEADS, SIGNAL_LEN
from src.basic.rule_ml import StepModule


def calc_output_shape(length_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    calculate the shape of the output from a convolutional/maxpooling layer
    """
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class EcgEmbed(StepModule):

    def __init__(self,
                 conv_out_channels=[32, 64, 16],
                 fc_out_dim=[512, 240],
                 conv_kernel_size=5,
                 conv_stride=1,
                 pool_kernel_size=2,
                 pool_stride=2):
        super().__init__()
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
        batched_ecg, batched_mid_output = x
        x = self.conv_layers(batched_ecg)
        x = x.view(-1, self.fc_input_dim)
        x = self.fc_layers(x)
        return [x, batched_mid_output]
