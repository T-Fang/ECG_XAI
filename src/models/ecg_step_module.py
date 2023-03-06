import torch
import torch.nn as nn
from src.basic.constants import PROB_THRESHOLD
from src.basic.rule_ml import StepModule
from src.basic.dx_and_feat import get_mid_output_by_str, fill_mid_output, Diagnosis


class BlockModule(StepModule):

    def __init__(self):
        super().__init__()
        self.PR_w = nn.Parameter(torch.ones(1), requires_grad=True)
        self.PR_delta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.QRS_w = nn.Parameter(torch.ones(1), requires_grad=True)
        self.QRS_delta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.NoNoModule = nn.Linear(1, 1)
        self.NoYesModule = nn.Linear(1, 1)
        self.YesNoModule = nn.Linear(1, 1)
        self.YesYesModule = nn.Linear(1, 1)

    def forward(self, x):
        batched_ecg_embed, batched_mid_output = x

        # Some logic operation here and update the mid_output

        for ecg_embed, mid_output in zip(batched_ecg_embed, batched_mid_output):
            LPR = get_mid_output_by_str(mid_output, 'LPR')
            LQRS = get_mid_output_by_str(mid_output, 'LQRS')

            if LPR > PROB_THRESHOLD and LQRS > PROB_THRESHOLD:
                pred = self.YesYesModule(ecg_embed)  # contains predicted probability for AVB, LBBB, RBBB
                fill_mid_output(mid_output, values=pred, keys=['AVB', 'LBBB', 'RBBB'], enum_type=Diagnosis)
            # etc.

        return (batched_ecg_embed, batched_mid_output)
