import tarfile
import os
from normalization import *
tar_files = [11] #[0,1,2,3,4,5,6,7,8,9,10,11]

####################
# Extract all files
####################

for tar_file in tar_files:
    my_tar = tarfile.open('normalized_data/' + tar_file  + '.tar')
    my_tar.extractall('normalized_data') # specify which folder to extract to
    my_tar.close()


####################
# Convert all files
####################

for tar_file in tar_files:
    files = os.listdir("normalized_data/"+ tar_file)
    path = "normalized_data/" + tar_file + "_normalized"
    os.mkdir(path)
    for file in files:
        file_path = "normalized_data/" +  tar_file + "/"+ file
        norm_path = 'normaized_data/' + tar_file + '_normalized' + '/' + file
        normalizeStaining(file_path, saveFile=norm_path, Io=240, alpha=1, beta=0.15)





####################
# Convert all files
####################


#tar = tarfile.open("sample.tar", "w")
#names = os.listdir("normalized_data/11")
#for name in names:
#    tar.add("normalized_data/11/" + name)
#tar.close()