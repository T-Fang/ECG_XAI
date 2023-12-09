import os
import ast
import pandas as pd


# if __name__ == '__main__':
def rm_main(data):
    # features = data.data.loc[data['Name'] == 'Summary', 'feature'].values[0]
    # midOutputs = data.iloc[0][]
    features = []
    midOutputs = []
    for _,row in data.iterrows():
        features += ast.literal_eval(row['Required Features'])
        midOutputs += ast.literal_eval(row['obj_feat_names'])

    features += midOutputs
    print(features)
    features=list(set(features))
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
