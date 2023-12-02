import os
import ast
import pandas as pd


# if __name__ == '__main__':
def rm_main(data):
    additional_diagnoses = data.loc[data['Name'] == 'Summary', 'diagnosis'].values[0]
    print(additional_diagnoses)
    additional_diagnoses = ast.literal_eval(additional_diagnoses)
    additional_diagnoses.sort()
    # additional_diagnoses = ["NORM", "AFIB", "AFLT", "SARRH", "SBRAD", "SR", "STACH", "AVB", "IVCD", "LAFB", "LBBB",
    #                         "LPFB", "RBBB", "WPW", "LAE", "LVH", "RAE", "RVH", "AMI", "IMI", "LMI"]
    diagnosis_code = '''import torch\nimport pandas as pd\nfrom enum import Enum\nclass Diagnosis(Enum):\n'''
    diagnosis_code += f"    NORM = 0\n"

    for index, diagnosis in enumerate(additional_diagnoses):
        diagnosis_code += f"    {diagnosis} = {index+1}\n"

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
