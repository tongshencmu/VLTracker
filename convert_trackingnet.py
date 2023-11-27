import glob
import sys
import os

exp_name = sys.argv[1]

input_folder = f'./work_dirs/test/tracking_results/ostrack/{exp_name}/trackingnet/'
output_folder = f'./work_dirs/test/tracking_results/ostrack/{exp_name}/trackingnet_converted/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for txt_file in glob.glob(input_folder + '*.txt'):
    
    txt_name = os.path.basename(txt_file)
    if 'time' in txt_name:
        continue
    
    input = open(txt_file, 'r')
    output = open(output_folder + txt_name, 'w')
    for line in input:
        new_line = line.replace('\t', ',')
        output.write(new_line)
    
    input.close()
    output.close()