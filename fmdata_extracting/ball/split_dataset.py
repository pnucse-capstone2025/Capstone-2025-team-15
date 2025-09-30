import os
import shutil
import random
from tqdm import tqdm

# 1. 경로 설정
# 원본 이미지와 라벨 파일이 있는 폴더
source_images_dir = "./source_data/train/images"
source_labels_dir = "./source_data/train/labels"

# 새로 데이터셋을 구성할 폴더
root_dir = "./dataset_folder" 

# 2. 분할 비율 설정
split_ratio = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

# 3. 폴더 생성
for split in split_ratio.keys():
    os.makedirs(os.path.join(root_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, split, 'labels'), exist_ok=True)

# 4. 파일 목록 가져오고 섞기
# 원본 이미지 폴더에서 이미지 파일 목록을 가져옵니다.
all_images = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_images) # 파일을 무작위로 섞음

# 5. 파일 분할 및 복사
total_files = len(all_images)
train_end = int(total_files * split_ratio['train'])
valid_end = train_end + int(total_files * split_ratio['valid'])

datasets = {
    'train': all_images[:train_end],
    'valid': all_images[train_end:valid_end],
    'test': all_images[valid_end:]
}

for split, file_list in datasets.items():
    print(f"Copying {split} files...")
    for filename in tqdm(file_list):
        basename = os.path.splitext(filename)[0]
        
        # 원본 이미지와 라벨 파일 경로
        img_src = os.path.join(source_images_dir, filename)
        label_src = os.path.join(source_labels_dir, basename + '.txt')

        # 대상 이미지와 라벨 파일 경로
        img_dst = os.path.join(root_dir, split, 'images', filename)
        label_dst = os.path.join(root_dir, split, 'labels', basename + '.txt')
        
        # 라벨 파일이 존재하는지 확인 후 이미지와 라벨을 함께 복사
        if os.path.exists(label_src):
            shutil.copyfile(img_src, img_dst)
            shutil.copyfile(label_src, label_dst)
        else:
            print(f"Warning: Label file for {filename} not found in {source_labels_dir}. Skipping.")

print("\nDataset splitting complete! 👍")
print(f"Total: {total_files} | Train: {len(datasets['train'])} | Valid: {len(datasets['valid'])} | Test: {len(datasets['test'])}")