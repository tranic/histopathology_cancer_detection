import tarfile
import os
from normalization import *
import time
tar_files = [11] #[0,1,2,3,4,5,6,7,8,9,10,11]

for_server = True

####################
# Extract all files
####################

if for_server:
    files_test = os.listdir("../data/test/")

    start_time_test = time.time()
    for file in files_test: 
        if files_test.index(file) % 1000 == 0:
            print("Normalizing: " + str(files_test.index(file)) + " out of " + str(len(files_test)))
        input_path = "../data/test/" + file
        output_path = "../data/test_normalized/" + file
        normalizeStaining(np.array(Image.open(input_path)), saveFile=output_path, Io=240, alpha=1, beta=0.15)
    end_time_test = time.time()

    

    files_train = os.listdir("../data/train/")

    start_time_train = time.time()
    for file in files_train: 
        if files_train.index(file) % 1000 == 0:
            print("Normalizing: " + str(files_train.index(file)) + " out of " + str(len(files_test)))
        input_path = "../data/train/" + file
        output_path = "../data/train_normalized/" + file
        normalizeStaining(np.array(Image.open(input_path)), saveFile=output_path, Io=240, alpha=1, beta=0.15)
    end_time_train = time.time()

    print("Normalization Time for test-data: --- %s seconds ---" % (end_time_test - start_time_test))
    print("Normalization Time for train-data: --- %s seconds ---" % (end_time_train - start_time_train))

else: 

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