import json

def convert_notebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Simple global replacements
            source = source.replace("medical_insurance.csv", "insdata100k.csv")
            source = source.replace("'charges'", "'annual_medical_cost'")
            source = source.replace('"charges"', '"annual_medical_cost"')
            source = source.replace(".charges", ".annual_medical_cost")
            source = source.replace("'children'", "'dependents'")
            source = source.replace('"children"', '"dependents"')
            
            # If pd.read_csv in source, drop person_id
            if 'pd.read_csv("insdata100k.csv")' in source and 'person_id' not in source:
                source = source.replace('pd.read_csv("insdata100k.csv")', 'pd.read_csv("insdata100k.csv")\nif "person_id" in data.columns:\n    data.drop(columns=["person_id"], inplace=True)')
            
            # Specific structure replacements
            if "for col in ['sex','smoker','region']:" in source and "label_encoder" in source:
                source = source.replace("for col in ['sex','smoker','region']:", "for col in data_corr.select_dtypes(include='object').columns:")
                
            if "dum = pd.get_dummies(data[['sex','region','smoker']],dtype=int)" in source:
                source = source.replace("pd.get_dummies(data[['sex','region','smoker']],dtype=int)", "pd.get_dummies(data.select_dtypes(include='object'), drop_first=True, dtype=int)")
                
            if "data[['age','bmi','dependents','annual_medical_cost']]" in source:
                source = source.replace("data[['age','bmi','dependents','annual_medical_cost']]", "data.select_dtypes(exclude='object')")
            
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'] = [x.replace('\n\n', '\n') for x in cell['source']]
            if len(cell['source']) > 0 and cell['source'][-1].endswith('\n') and source and not source.endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

convert_notebook("meds_tuned.ipynb", "insdata100k_tuned.ipynb")
