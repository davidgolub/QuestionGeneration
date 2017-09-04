import zipfile
import glob
import datetime
import os
import urllib
import shutil
import os
import pickle

def pickle_save(data, save_path):
	print("Saving data to path %s" % save_path)
	save_file = open(save_path, 'w')
	pickle.dump(data, save_file)
	save_file.close()

def pickle_load(load_path):
	print("Loading data from path %s" % load_path)

	load_file = open(load_path, 'r')
	data = pickle.load(load_file)
	load_file.close()

	print("Done loading data from path %s" % load_path)
	return data

def get_subdirs(src_dir):
	return [os.path.join(src_dir, name) for name in os.listdir(src_dir) \
	if os.path.isdir(os.path.join(src_dir, name))]

def copy_files(src_dir, dest_dir):
	for filename in glob.glob(os.path.join(src_dir, '*.*')):
		shutil.copy(filename, dest_dir)

def copy_file(src_name, dest_name):
	shutil.copyfile(src_name, dest_name)

def get_files(src_dir):
	"""
	Gets all files from source directory
	"""
	files = glob.glob(os.path.join(src_dir, '*.*'))
	return files

def download_file(url, save_path):
	""" Downloads url to save_path """
	url_opener = urllib.URLopener()
	url_opener.retrieve(url, save_path)

def check_dir(save_dir):
	""" Creates dir if not exists"""
	if not os.path.exists(save_dir):
		print("Directory %s does not exist, making it now" % save_dir)
		os.makedirs(save_dir)
		return False
	else:
		print("Directory %s exists, all good" % save_dir)
		return True

def get_matching_files(regex):
	files = glob.glob(regex)
	return files

def zip_files(file_list, save_path):
	print('creating archive into path %s' % save_path)
	zf = zipfile.ZipFile(save_path, mode='w')

	for f in file_list:
		print(f)
		zf.write(f)
	zf.close()
	print_info(save_path)

def unzip_files(zip_path, directory_to_extract_to):
	print("Unzipping files from path %s to dir %s" \
		% (zip_path, directory_to_extract_to))
	zip_ref = zipfile.ZipFile(zip_path, 'r')
	zip_ref.extractall(directory_to_extract_to)
	zip_ref.close()

def print_info(archive_name):
    zf = zipfile.ZipFile(archive_name)
    for info in zf.infolist():
        print(info.filename)
        print('\tComment:\t', info.comment)
        print('\tModified:\t', datetime.datetime(*info.date_time))
        print('\tSystem:\t\t', info.create_system, '(0 = Windows, 3 = Unix)')
        print('\tZIP version:\t', info.create_version)
        print('\tCompressed:\t', info.compress_size, 'bytes')
        print('\tUncompressed:\t', info.file_size, 'bytes')
        print

