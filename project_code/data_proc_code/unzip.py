import os, zipfile
import shutil
import tarfile,gzip

dir_name = '/Users/jessica/Documents/masterproject/dataset_Sourceforge'

os.chdir(dir_name) # change directory from working dir to dir with files

for item in os.listdir(dir_name): # loop through items in dir
    
    if item.endswith(".zip"): # check for ".zip" extension
        save_dir = dir_name+'/'+item[:-4]
        os.makedirs(save_dir)
        file_name = os.path.abspath(item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall(save_dir) # extract file to dir,unzip
        zip_ref.close() # close file
        #os.remove(file_name) # delete zipped file

    if item.endswith(".tar.gz") or item.endswith(".gz"):
        if item.endswith(".tar.gz"): 
            save_dir = dir_name+'/'+item[:-7]
        elif item.endswith(".gz"):
            save_dir = dir_name+'/'+item[:-3]
        
        file_name = os.path.abspath(item) 
        try:#one of the .gz file can not be opened with tarfile.open()
            g_file = tarfile.open(file_name) 
            os.makedirs(save_dir)
            g_file.extractall(save_dir)
            g_file.close()
        except:
            continue 

# TODO: no such file on my computer, still need to be validated.
    # elif item.endswith(".tar"):
    #     save_dir = dir_name+'/'+item[:-3]
    #     os.makedirs(save_dir)
    #     file_name = os.path.abspath(item) 
    #     tar = tarfile.open(file_name) 
    #     names = tar.getnames()
    #     for name in names:
    #         tar.extract(name,save_dir) 
    #     zip_ref.close() # close file
    #     #os.remove(file_name) # delete zipped file

    # elif item.endswith(".rar"):
    #     save_dir = dir_name+'/'+item[:-3]
    #     os.makedirs(save_dir)
    #     file_name = os.path.abspath(item) 
    #     rar = rarfile.RarFile(file_name) 
    #     rar.extractall(save_dir)
    #     rar.close() # close file
    #     #os.remove(file_name) 

    
#extrat all the .exe file from the dataset file
exe_save_path = '/Users/jessica/Documents/masterproject/exefiles_sourceforge'
for curDir, dirs, files in os.walk(dir_name):
    for file in files:
        if file.endswith(".exe"):
            file_path = os.path.join(curDir, file)
            shutil.move(file_path, exe_save_path)
