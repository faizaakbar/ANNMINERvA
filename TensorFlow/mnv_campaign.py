"""
python mnv_campaign.py MODEL_CODE_BASE
"""
from __future__ import print_function
import subprocess
import ConfigParser
import time
import os
import sys


if '-h' in sys.argv or '--help' in sys.argv:
    print(__doc__)
    sys.exit(1)

if not len(sys.argv) == 2:
    print('The model code base is mandatory.')
    print(__doc__)
    sys.exit(1)

model_code_base = str(sys.argv[1])

config = ConfigParser.ConfigParser()
config.read('./mnv_st_epsilon.cfg')
# config.sections()
# config.options('RunOpts')
# config.get('RunOpts', 'short')

p = subprocess.Popen('hostname', shell=True, stdout=subprocess.PIPE)
host_name = p.stdout.readlines()[0].strip()

repo_info_string = """
# print identifying info for this job
cd /home/perdue/ANNMINERvA/TensorFlow/WilsonCluster
echo "Workdir is `pwd`"
GIT_VERSION=`git describe --abbrev=12 --dirty --always`
echo "Git repo version is $GIT_VERSION"
DIRTY=`echo $GIT_VERSION | perl -ne 'print if /dirty/'`
if [[ $DIRTY != "" ]]; then
  echo "Git repo contains uncomitted changes!"
  echo ""
  echo "Changed files:"
  git diff --name-only
  echo ""
  # exit 0
fi
"""

tstamp = str(int(time.time()))
job_name = 'blah' + tstamp + '.sh'
arg_parts = []

# run opts
log_level = config.get('RunOpts', 'log_level')
num_epochs = int(config.get('RunOpts', 'num_epochs'))

# data description
n_classes = int(config.get('DataDescription', 'n_classes'))
n_planecodes = int(config.get('DataDescription', 'n_planecodes'))
imgw_x = int(config.get('DataDescription', 'imgw_x'))
imgw_uv = int(config.get('DataDescription', 'imgw_uv'))
targets_label = config.get('DataDescription', 'targets_label')
filepat = config.get('DataDescription', 'filepat')
compression = config.get('DataDescription', 'compression')

# sample labels
train_sample = config.get('SampleLabels', 'train')
valid_sample = config.get('SampleLabels', 'valid')
test_sample = config.get('SampleLabels', 'test')
pred_sample = config.get('SampleLabels', 'pred')

# training opts - batch_norm is used in multiple places
optimizer = config.get('Training', 'optimizer')
batchf = 'do_batch_norm' if int(config.get('Training', 'batch_norm')) > 0 \
         else 'nodo_batch_norm'
batch_size = int(config.get('Training', 'batch_size'))

# model code
model_code = model_code_base + '_' + host_name + '_' + str(batch_size) + \
             '_' + optimizer + '_train' + train_sample + '_valid' + \
             valid_sample + '_' + batchf + '_' + targets_label + str(n_classes)

# paths
basep = config.get('Paths', 'basep')
data_basep = os.path.join(
    basep,
    config.get('Paths', 'data_path_ext'),
    config.get('Paths', 'processing_version')
)
data_dirs = '--data_dir ' + ','.join(
    [os.path.join(data_basep, pth)
     for pth in config.get('Paths', 'data_ext_dirs').split(',')]
)
log_dir = '--log_name ' + os.path.join(
    basep,
    config.get('Paths', 'log_path_ext'),
    config.get('Paths', 'processing_version'),
    'log_mnv_st_epsilon_' + model_code + '_' + tstamp + '.txt'
)
model_dir = '--model_dir ' + basep + '/models/' + str(n_classes) + \
            '/' + model_code
pred_file = '--pred_store_name ' + os.path.join(
    basep,
    config.get('Paths', 'pred_path_ext'),
    'mnv_st_epsilon_predictions' + pred_sample + '_model_' + model_code
)

# singularity
container = config.get('Singularity', 'container')

arg_parts.append('--n_planecodes %d' % n_planecodes)
arg_parts.append('--n_classes %d' % n_classes)
arg_parts.append('--imgw_x %d --imgw_uv %d' % (imgw_x, imgw_uv))
arg_parts.append('--targets_label %s' % targets_label)
if compression in ['gz', 'zz']:
    arg_parts.append('--compression %s' % compression)
arg_parts.append('--file_root ' + filepat + str(imgw_x) + '_')

if optimizer is not '':
    arg_parts.append('--strategy %s' % optimizer)
arg_parts.append('--batch_size %d' % batch_size)
arg_parts.append('--%s' % batchf)

arg_parts.append(data_dirs)
arg_parts.append(log_dir)
arg_parts.append(model_dir)

# run opt switches
arg_parts.append('--log_level %s' % log_level)
arg_parts.append('--num_epochs %d' % num_epochs)
switches = ['training', 'validation', 'testing', 'prediction',
            'use_all_for_test', 'use_test_for_train', 'use_valid_for_test',
            'a_short_run', 'log_devices']
for switch in switches:
    arg_parts.append(
        '--do_{}'.format(switch)
        if int(config.get('RunOpts', switch))
        else '--nodo_{}'.format(switch)
    )
if '--do_prediction' in arg_parts:
    arg_parts.append(pred_file)

arg_string = ' '.join(arg_parts)

run_script = config.get('Code', 'run_script')

with open(job_name, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('echo "started "`date`" "`date +%s`""\n')
    if 'gpu' in host_name:
        f.write('nvidia-smi -L\n')
    f.write(repo_info_string)
    if container is not '':
        f.write('singularity exec {} python {} {}\n'.format(
            container, run_script, arg_string
        ))
    else:
        f.write('python {} {}'.format(
            run_script, arg_string
        ))
    f.write('nvidia-smi -L >> $LOGFILE\n')
    f.write('nvidia-smi >> $LOGFILE\n')
