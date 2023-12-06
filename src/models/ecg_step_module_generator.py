import os
import ast
import pandas as pd
import re


# if __name__ == '__main__':
def rm_main(data):
    
    step2 = data.iloc[1]
    name = step2['Name']
    focused_leads = ast.literal_eval(step2['focused leads'])
    obj_feat_names = ast.literal_eval(step2['obj_feat_names'])
    thresholds = ast.literal_eval(step2['thresholds'])
    comp_op_names = ast.literal_eval(step2['comp_op_names'])
    norm_if_not = ast.literal_eval(step2['NORM_if_NOT'])
    traces = ast.literal_eval(step2['traces'])
    operations = ast.literal_eval(step2['Operations'])
    required_features = ast.literal_eval(step2['Required Features'])
    result_outputs = ast.literal_eval(step2['ResultOutputs'])

    for key,value in thresholds.items():
        thresholds[key]=int(''.join(filter(str.isdigit, value)))

    print('name:', name)
    print('focused_leads:', focused_leads)
    print('obj_feat_names:', obj_feat_names)
    print('thresholds:', thresholds)
    print('comp_op_names:', comp_op_names)
    print('norm_if_not:', norm_if_not)
    print('traces:', traces)
    print('operations:', operations)
    print('required_features:', required_features)
    print('result_outputs:', result_outputs)

    code = ("import torch\nimport torch.nn as nn\nimport pandas as pd\nfrom src.basic.rule_ml import LT, And, Or, "
            "PipelineModule, StepModule, SeqSteps, Imply, GT, Not, ComparisonOp, get_agg_col_name\nfrom "
            "src.basic.dx_and_feat import Diagnosis, Feature, get_by_str\nfrom src.models.ecg_step_module import "
            "EcgStep\n")

    code += f"\nclass {name}_Module(EcgStep):\n"
    code += f"    focused_leads: list[str] = {focused_leads}\n"
    code += f"    obj_feat_names: list[str] = {obj_feat_names}\n"
    code += f"    feat_imp_names: list[str] = {[x + '_imp' for x in obj_feat_names]}\n"
    code += f"    comp_op_names: list[str] = {comp_op_names}\n"
    code += f"    imply_names: list[str] = {[x + '_imply' for x in result_outputs]}\n"
    code += f"    pred_dx_names: list[tuple[str, list[str]]] = {[('NORM', ['NORM_imp'])] + [(x, [x + '_imp']) for x in result_outputs]}\n"
    code += f"    NORM_if_NOT: list[str] = {norm_if_not}\n"

    code += f"\n    def __init__(self, all_mid_output: dict[str, dict[str, torch.Tensor]], hparams, is_using_hard_rule: bool = False):\n"
    code += f"        super().__init__(self.module_name, all_mid_output, hparams, is_using_hard_rule)\n"

    code += f"\n        # Comparison operators\n"
    for i in range(len(obj_feat_names)):
        code += f"        self.{comp_op_names[i]} = " + (
            "LT" if comp_op_names[i][-2:] == 'lt' else 'GT') + f"(self, '{obj_feat_names[i]}', {thresholds[obj_feat_names[i]]})\n"

    code += f"\n        # Imply\n"
    for i in range(len(result_outputs)):
        code += f"        self.{result_outputs[i]}_imply = Imply(self, self.get_imply_hparams(hparams, '{traces[i][:-len(result_outputs[i])]}_imp', ['{result_outputs[i]}_imp']))\n"

    code += f"\n    def apply_rule(self, x) -> None:\n"
    code += f"        batched_ecg, batched_obj_feat = x\n"

    code += f"\n        # Extract objective features: {', '.join(obj_feat_names)}\n"
    for i in range(len(required_features)):
        code += f"        {required_features[i]} = get_by_str(batched_obj_feat, ['{required_features[i]}'], Feature)\n"

    code += f"\n        # Comparison operators\n"
    for i in range(len(required_features)):
        code += f"        self.{comp_op_names[i]}({required_features[i]})\n"

    code += f"\n        # Imply\n"
    code += f"        focused_embed = self.get_focused_embed()\n"
    code += f"        decision_embed = None if self.use_lattice else self.imply_decision_embed_layer(focused_embed)\n"
    code += f"        imply_input = (focused_embed, decision_embed)\n"

    for i in range(len(traces)):
        code += f"\n        # {traces[i]}\n"
        code += f"        self.{result_outputs[i]}_imply(imply_input)\n"

    code += f"\n    def add_explanation(self, mid_output_agg: pd.Series, report_file_obj):\n"
    code += f"        report_file_obj.write('## Step 2: {name} Module\\n')\n"
    for i in range(len(result_outputs)):
        code += f"        self.add_imply_exp(mid_output_agg, report_file_obj, '{traces[i]}', '', '{result_outputs[i]}_imp')\n"

    for i in range(len(obj_feat_names)):
        code += f"        self.add_comp_exp(mid_output_agg, report_file_obj, '{comp_op_names[i]}')\n"


    file_name = "ecg_step_module2.py"
    with open(file_name, "w") as file:
        file.write(code)
