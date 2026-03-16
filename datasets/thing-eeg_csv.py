import csv
from pathlib import Path
import re

def extract_id_category(name: str):
    """
    get (4, 'abcde') from '00004_abcde'
    return None if the dir didn`t exist
    """
    m = re.fullmatch(r'(\d{5})_(.+)', name)
    return (int(m.group(1)), m.group(2)) if m else None

def folders_to_csv(root_dir: str, csv_path: str):
    """
    root_dir : the path of dir which will be scanned
    csv_path : the path of output csv
    """
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        raise ValueError(f'{root} didn`t exist')

    rows = []
    for item in root.iterdir():
        if item.is_dir():
            res = extract_id_category(item.name)
            if res:
                rows.append(res)

    rows.sort(key=lambda x: x[0])         
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(f'{"id":<8}{"category":<20}\n')
        for idx, cat in rows:
            f.write(f'{idx:<8}{cat:<20}\n')

if __name__ == '__main__':
    scan_folder = r'/mnt/dataset2/Datasets/Things-EEG2/training_images/training_images'   # input
    out_csv     = r'/mnt/dataset2/Processed_datasets/EEG_Bench/Thing_EEG2_training/category.csv'    # output
    folders_to_csv(scan_folder, out_csv)
    print('Finished. The output is in ', out_csv)