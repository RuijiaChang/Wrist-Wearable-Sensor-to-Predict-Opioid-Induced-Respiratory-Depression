import pandas as pd
import os
# three methods are proposed: 1.simply time shift by 1
#                             2.predict with 3 past data sets
#                             3.predict with weighted sum of past 3 data sets

# method 1:
def Use_past_1(input_csv: str, output_dir: str, outputName: str="BIDMC_features_Use_P1.csv") -> None:
    # unchanged elements
    target_set = ["SpO2(mean)", "RR(mean)", "wave nunmber", "segment nunmber"]
    # 31 segments per wave
    block_length = 31
    dir_path = os.path.join(output_dir, outputName)
    df = pd.read_csv(input_csv)
    # features columns 
    features = [feature for feature in df.columns if feature not in target_set]
    # store each processed segment
    result = []
    for i in range(54):
        # isolate each sement
        start = i*block_length
        end = start + block_length
        curr_seg = df.iloc[start:end]

        # shift features columns down by 1, drow Na rows (ie. 1st row) 
        curr_seg.loc[:, features] = curr_seg[features].shift(1)
        curr_seg = curr_seg.dropna()
        
        result.append(curr_seg)

    result_csv = pd.concat(result, ignore_index=True)
    result_csv.to_csv(dir_path, index=False)

# method 2:
def Use_past_3(input_csv: str, output_dir: str) -> None:
    target_set = ["SpO2(mean)", "RR(mean)", "wave nunmber", "segment nunmber"]
    block_length = 31
    dir_path = os.path.join(output_dir, "BIDMC_features_Use_P3.csv")
    df = pd.read_csv(input_csv)
    features = [feature for feature in df.columns if feature not in target_set]
    
    result = []
    for i in range(54):
        start = i*block_length
        end = start + block_length
        curr_seg = df.iloc[start:end]
        
        # consider 3 past feature sets
        past_num = 3

        # store past feature columns
        past_cols = {}
        for col in features:
            for past in range(1, past_num + 1):
                past_cols[f"{col}_p{past}"] = curr_seg[col].shift(past)

        curr_seg = pd.concat([curr_seg, pd.DataFrame(past_cols)], axis=1)

        # discard original columns and Na rows
        curr_seg.drop(columns=features, inplace=True)
        curr_seg = curr_seg.dropna()

        result.append(curr_seg)
    
    result_csv = pd.concat(result, ignore_index=True)
    result_csv.to_csv(dir_path, index=False)

def Use_weighted_P3(input_csv: str, output_dir: str, weights: list[float] = [0.33, 0.33, 0.33], outputName: str="BIDMC_features_Use_PW3.csv") -> None:
    target_set = ["SpO2(mean)", "RR(mean)", "wave nunmber", "segment nunmber"]
    # weights = [0.5, 0.3, 0.2]
    # weights = [0.33, 0.33, 0.33]
    block_length = 31
    dir_path = os.path.join(output_dir, outputName)
    df = pd.read_csv(input_csv)
    features = [feature for feature in df.columns if feature not in target_set]
    
    result = []
    for i in range(54):
        start = i*block_length
        end = start + block_length
        curr_seg = df.iloc[start:end]

        for col in features:
            # calculate weighted average
            weighted_sum =(curr_seg[col].shift(1)*weights[0] 
                         + curr_seg[col].shift(2)*weights[1]
                         + curr_seg[col].shift(3)*weights[2])
            curr_seg.loc[:,col] = weighted_sum
        
        curr_seg = curr_seg.iloc[3:]

        result.append(curr_seg)

    result_csv = pd.concat(result, ignore_index=True)
    result_csv.to_csv(dir_path, index=False)

def Use_weighted_C3(input_csv: str, output_dir: str, weights: list[float] = [0.33, 0.33, 0.33], outputName: str="BIDMC_features_Use_CW3.csv") -> None:
    target_set = ["SpO2(mean)", "RR(mean)", "wave nunmber", "segment nunmber"]
    # weights = [0.5, 0.3, 0.2]
    # weights = [0.4, 0.3, 0.3]
    block_length = 31
    dir_path = os.path.join(output_dir, outputName)
    df = pd.read_csv(input_csv)
    features = [feature for feature in df.columns if feature not in target_set]
    
    result = []
    for i in range(54):
        start = i*block_length
        end = start + block_length
        curr_seg = df.iloc[start:end]

        for col in features:

            weighted_sum =(curr_seg[col]*weights[0] 
                         + curr_seg[col].shift(1)*weights[1]
                         + curr_seg[col].shift(2)*weights[2])
            curr_seg.loc[:,col] = weighted_sum
        
        curr_seg = curr_seg.iloc[2:]

        result.append(curr_seg)

    result_csv = pd.concat(result, ignore_index=True)
    result_csv.to_csv(dir_path, index=False)

def main():
    input_csv = "BIDMC_Regression/features/BIDMC_Segmented_features.csv"
    output_dir = "BIDMC_Regression/features"

    # Use_past_1(input_csv, output_dir, "BIDMC_Segmented_features_Use_P1.csv")
    # Use_past_3(input_csv, output_dir)
    Use_weighted_P3(input_csv, output_dir, weights=[0.242, 0.519, 0.240], outputName="BIDMC_Segmented_features_Use_PWOptuna.csv")
    # Use_weighted_C3(input_csv, output_dir, weights=[0.5, 0.3, 0.2], outputName="BIDMC_Segmented_features_Use_CW532.csv")
    return 0

if __name__ == '__main__':
    main()

