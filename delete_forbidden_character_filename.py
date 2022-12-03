import os 
list = ['testA', 'testB', 'testC', 'testD', 'testE', 'trainA', 'trainB', 'trainC','trainD', 'trainE']
for i in list:
    paths = (os.path.join(root, filename)
        for root, _, filenames in os.walk('/Users/macos/Desktop/GAN_Project/CycleGAN/final_data/'+i)
        for filename in filenames)

    for path in paths:
        newname = path.replace("'", '')
        if newname != path:
            os.rename(path, newname)
    print(f"{i} Successful !!!")