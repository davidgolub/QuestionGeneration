import subprocess
import time
import re
from subprocess import Popen
import urllib2, urllib
import json
import httplib
import os
import urlparse
from optparse import OptionParser

# This wraps a process within a logger and logs the information in the cloud
def spawner(cmd_list):
	print("Spawning process")
	p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	print("Done spawning")
	# Parameters needed for logging
	job_id = -1
	job_id_got = False

	job_endpoint = 'NONE'
	job_endpoint_got = False

	error_log_endpoint = 'NONE'
	error_log_endpoint_got = False

	weird_constant = '\x1b[0m' # For some reason adds this weird constant to every line

	while True:
	  print("Reading line out")
	  line = p.stdout.readline()
	  print(line)
	  if line != '':
	    #the real code does filtering here
	    stripped_line = line.rstrip()
	    stripped_line = str.replace(stripped_line, weird_constant, '', )

	    print('STDOUT ' + stripped_line)
	    if not job_id_got:
	    	print("Got job id")
	    	job_id_got, job_id = get_job_id(stripped_line)

	    if not job_endpoint_got:
	    	print("Got job endpoint")
	    	job_endpoint_got, job_endpoint = get_job_endpoint(stripped_line)

	    if not error_log_endpoint_got:
	    	print("Getting error log endpoint")
	    	error_log_endpoint_got, error_log_endpoint = get_error_log_endpoint(stripped_line)
	  else:
	    break

	stdout, stderr = p.communicate()

	if stderr is not None and stderr is not ' ':
		# Job url: http://api_endpoint/api/job/id/
		# job_endpoint: http://api_endpoint/api/job/

		print('STDERR ' + stderr)
		job_url = "{0}{1}{2}".format(job_endpoint, int(job_id), '/')

		print("JOB URL IS " + job_url)
		data = {}
		data['job'] = job_url
		data['text'] = stderr

		send_post_message(error_log_endpoint.rstrip(), data)
	else:
		print("Job finished successfully.")

def send_post_message(url, data):
	# split url into base and relative path
	result = urlparse.urlparse(url)
	base_path = result.netloc
	relative_path = result.path 
	encoded_data = urllib.urlencode(data)

	h = httplib.HTTPConnection(base_path)

	headers = {"Content-type": "application/x-www-form-urlencoded"}

	h.request('POST', '/api/v1/error_log/', encoded_data, headers)

	r = h.getresponse()
	print r.read()


def get_job_id(message):
	m = re.search('(?<=job with id )[0-9]*', message)
	if m is not None:
		job_id = m.group(0)
		print("Got job id %s" % job_id)
		return True, job_id
	else:
		return False, -1

def get_job_endpoint(message):
	m = re.search('(?<=Job endpoint ).*', message)
	if m is not None:
		print("Got job endpoint " + message)
		job_endpoint = m.group(0)
		print("Job endpoint %s" % job_endpoint)
		return True, job_endpoint
	else:
		return False, ''

def get_error_log_endpoint(message):
	m = re.search('(?<=Error log endpoint ).*', message)
	if m is not None:
		print("Got error log endpoint")
		error_endpoint = m.group(0)
		print("Endpoint is %s" % error_endpoint)
		return True, error_endpoint
	else:
		return False, ''

if __name__ == '__main__':
	print("Starting stuff")
	parser = OptionParser()
	parser.add_option("--command", "--command", dest="command",
	                  help="Command to execute")

	parser.add_option("--args", "--args", dest="args",
	                  help="Args for command")

	(options, args) = parser.parse_args()

	split_args = options.args.split(' ')

	concatenated_args = [options.command]
	concatenated_args.extend(split_args)
	spawner(concatenated_args)