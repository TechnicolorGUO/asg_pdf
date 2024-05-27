import pandas as pd

input_folder_dir = './'
input_file_name_list = ['2830555.tsv','2907070.tsv','3073559.tsv','3274658.tsv']
output_file_dir = './merge.tsv'

input_file_dir = input_folder_dir + '2742488.tsv'
merge_tsv = pd.read_csv(input_file_dir, sep='\t',header=0)

for input_file_name in input_file_name_list:
    input_file_dir = input_folder_dir + input_file_name
    input_tsv = pd.read_csv(input_file_dir, sep='\t',header=0)
    merge_tsv = pd.concat([merge_tsv,input_tsv], axis=0, sort=False)
    print(input_tsv.shape)

print(merge_tsv.shape)
merge_tsv.drop_duplicates('ref_title', inplace=True)
print(merge_tsv.shape)


merge_tsv.to_csv(output_file_dir, sep='\t', index = False)