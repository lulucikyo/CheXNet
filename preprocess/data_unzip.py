import tarfile

extract_path = "."
for i in range(1,13):
    filename = "images_"+("0"*(3-len(str(i))))+str(i)+".tar.gz"
    print(filename)
    tar = tarfile.open(filename, "r:gz")
    tar.extractall(extract_path)