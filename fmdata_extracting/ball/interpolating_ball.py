import pandas as pd
import numpy as np
import os



def interpolate_missing_positions(csv_path, output_dir=None):
    df = pd.read_csv(csv_path)

    # 보간 (선형)
    df['x'] = df['x'].interpolate(method='linear')
    df['y'] = df['y'].interpolate(method='linear')

    # (선택) 처음/끝 NaN → 앞/뒤 채우기
    df['x'] = df['x'].fillna(method='bfill')
    df['y'] = df['y'].fillna(method='bfill')

    output_path = "./ball/interpolation/balls_inter.csv"
    
    df.to_csv(output_path, index=False)
    print(f"보간된 결과가 {output_path}에 저장되었습니다.")

    return df

csv_file_name = "./ball/output/balls3.csv"
df = interpolate_missing_positions(csv_file_name)
    


