import os

basepath = '/Users/paulhsu/CIS537_Data'
for fname in os.listdir(basepath):
    path = os.path.join(basepath, fname)
    print (path)
    if os.path.isdir(path):
        continue