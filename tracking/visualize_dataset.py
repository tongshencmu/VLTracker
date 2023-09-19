import cv2
import glob
import numpy as np
import os


def draw_bounding_boxes(images_folder, bounding_boxes, output_video_path, fps=30):

    # Get a list of image file names in the folder
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Read the first image to get dimensions
    first_image_path = os.path.join(images_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    
    # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image_file, bounding_box_data in zip(image_files, bounding_boxes):
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        
        # Draw bounding boxes on the image
        x, y, w, h = bounding_box_data
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Write the image to the video
        video_writer.write(image)
    
    # Release the VideoWriter object
    video_writer.release()
    
dataset_path = "/ocean/projects/ele220002p/tongshen/dataset/lasot/LaSOTBenchmark/"
data_folders = glob.glob(dataset_path + "/*/")

result_path = '/ocean/projects/ele220002p/tongshen/code/vl_tracking/lasot_results/'
result_txts = glob.glob(result_path + "*.txt")

with open('example_seqs.txt') as f:
    lines = f.readlines()
    
example_seqs = [ll.strip() for ll in lines]
print(example_seqs)

for folder in data_folders:
    
    category = folder.split('/')[-2]
    
    seqs = glob.glob(folder + "/*/")
    for seq in seqs:
        seq_name = seq.split('/')[-2]
        
        if seq_name not in example_seqs:
            continue
        
        result_txt = result_path + seq_name + '.txt'
        if not os.path.isfile(result_txt):
            continue
        
        with open(result_txt) as f:
            lines = f.readlines()
            
        bboxes = [list(map(int, ll.split('\t'))) for ll in lines]
            
        images = glob.glob(seq + "/img/*.*")

        if os.path.isfile(f"lasot_visuals_clean/{seq_name}.avi"):
            continue
        draw_bounding_boxes(seq + "/img/", bboxes, f"./work_dirs/example_seqs/{seq_name}.avi")
        
        print(seq_name)
