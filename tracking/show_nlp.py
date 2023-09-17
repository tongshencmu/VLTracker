import numpy as np

with open('/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/VLTracker/tracking/hard_seqs.txt') as f:
    lines = f.readlines()
    
for line in lines:
    
    line = line.strip()
    cate_name = line.split('-')[0]
    nlp_path = f'/ocean/projects/ele220002p/tongshen/dataset/lasot/LaSOTBenchmark/{cate_name}/{line}/nlp.txt'
    
    try:
        with open(nlp_path) as f:
            lines = f.readlines()
            print(lines[0].strip())
    except:
        print(nlp_path)