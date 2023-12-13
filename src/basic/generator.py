import os
import ast
import pandas as pd


# if __name__ == '__main__':

def module_generation(data):
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

    for key, value in thresholds.items():
        thresholds[key] = int(''.join(filter(str.isdigit, value)))

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
            "LT" if comp_op_names[i][
                    -2:] == 'lt' else 'GT') + f"(self, '{obj_feat_names[i]}', {thresholds[obj_feat_names[i]]})\n"

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

    file_name = "../models/ecg_step_module2.py"
    with open(file_name, "w") as file:
        file.write(code)


def feat_generation(data):
    features = []
    midOutputs = []
    for _, row in data.iterrows():
        features += ast.literal_eval(row['Required Features'])
        midOutputs += ast.literal_eval(row['obj_feat_names'])

    features += midOutputs
    print(features)
    features = list(set(features))
    features.sort()
    # features = ["NORM", "AFIB", "AFLT", "SARRH", "SBRAD", "SR", "STACH", "AVB", "IVCD", "LAFB", "LBBB",
    #                         "LPFB", "RBBB", "WPW", "LAE", "LVH", "RAE", "RVH", "AMI", "IMI", "LMI"]
    feature_code = '''import torch\nimport pandas as pd\nfrom enum import Enum\nclass Feature(Enum):\n'''
    index = 0
    for feature in features:
        if "(lead)" in feature:
            for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
                lead = "_" + lead
                feature_code += f"    {feature.replace('(lead)', lead)} = {index}\n"
                index += 1
        else:
            feature_code += f"    {feature} = {index}\n"
            index += 1

    file_name = "features.py"

    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(feature_code)
    else:
        with open(file_name, "r") as file:
            content = file.read()

        insert_position = content.find("class Feature(Enum):")
        content = content[:insert_position] + feature_code[insert_position:]
        with open(file_name, "w") as file:
            file.write(content)


def dx_generation(data):
    additional_diagnoses = []
    for _, row in data.iterrows():
        additional_diagnoses += ast.literal_eval(row['diagnosis'])

    additional_diagnoses = list(set(additional_diagnoses))
    additional_diagnoses.sort()
    print(additional_diagnoses)

    # additional_diagnoses = ["NORM", "AFIB", "AFLT", "SARRH", "SBRAD", "SR", "STACH", "AVB", "IVCD", "LAFB", "LBBB",
    #                         "LPFB", "RBBB", "WPW", "LAE", "LVH", "RAE", "RVH", "AMI", "IMI", "LMI"]
    diagnosis_code = '''import torch\nimport pandas as pd\nfrom enum import Enum\nclass Diagnosis(Enum):\n'''
    diagnosis_code += f"    NORM = 0\n"

    for index, diagnosis in enumerate(additional_diagnoses):
        diagnosis_code += f"    {diagnosis} = {index + 1}\n"

    file_name = "diagnosis.py"

    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(diagnosis_code)
    else:
        with open(file_name, "r") as file:
            content = file.read()

        insert_position = content.find("class Diagnosis(Enum):")
        content = content[:insert_position] + diagnosis_code[insert_position:]
        with open(file_name, "w") as file:
            file.write(content)


def rm_main(data):
    module_generation(data)
    feat_generation(data)
    dx_generation(data)
