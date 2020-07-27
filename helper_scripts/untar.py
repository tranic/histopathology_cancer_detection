import tarfile

my_tar = tarfile.open('../data/not_normalized_data/0.tar')
my_tar.extractall('../data/normalized_data/') # specify which folder to extract to
my_tar.close()