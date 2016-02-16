"""
Usage:
    python plot_valid.py WilsonCluster/saved_logs/<log file name>
"""
from __future__ import print_function
import re
import sys
import numpy as np
import pylab

logname = sys.argv[1]

git_repo_match = re.compile(r"^Git repo version is")
learning_rate_match = re.compile(r"^ Learning rate:")
momentum_match = re.compile(r"^ Momentum:")
datasize_match = \
    re.compile(r"^Learning data size: ")
epoch_match = re.compile(r"^Epoch [0-9]+ of [0-9]+ took")
val_acc_match = re.compile(r"^  validation accuracy:")
val_loss_match = re.compile(r"^  validation loss:")

epochs = []
val_accs = []
val_loss = []
run_times = []

tstamp = int(re.findall(r"[0-9]+", logname)[0])
print("Log timestamp =", tstamp)

with open(logname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        gt_m = re.search(git_repo_match, line)
        lr_m = re.search(learning_rate_match, line)
        mm_m = re.search(momentum_match, line)
        ds_m = re.search(datasize_match, line)
        ep_m = re.search(epoch_match, line)
        va_m = re.search(val_acc_match, line)
        vl_m = re.search(val_loss_match, line)
        if gt_m:
            repo = line.split()[-1]
            print("repo hash =", repo)
        if lr_m:
            lr = float(line.split(':')[-1])
            print("learning rate =", lr)
        if mm_m:
            mm = float(line.split(':')[-1])
            print("momentum = ", mm)
        if ds_m:
            nums_m = re.findall(r"[0-9]+", line)
            num_learn = int(nums_m[0])
            print(num_learn)
            img_w = int(nums_m[2])
            img_h = int(nums_m[3])
            print("img w, h = ", img_w, ",", img_h)
        if ep_m:
            nums_e = re.findall(r"[0-9]+", line)
            epochs.append(int(nums_e[0]))
            runt = float(str(nums_e[2] + '.' + nums_e[3]))
            run_times.append(runt)
        if va_m:
            val_accs.append(float(line.split()[-2]))
        if vl_m:
            val_loss.append(float(line.split()[-1]))

epochs = np.asarray(epochs)
val_accs = np.asarray(val_accs)
val_loss = np.asarray(val_loss)
run_times = np.asarray(run_times)

fig = pylab.figure(figsize=(5, 5))
pylab.title('Validation accuracy vs epoch')
pylab.plot(epochs, val_accs)
pylab.xlabel('epoch')
pylab.ylabel('validation accuracy (%)')
pylab.annotate(repo,
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.5), textcoords='figure fraction')
pylab.annotate("Learning rate = %f" % (lr),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.45), textcoords='figure fraction')
pylab.annotate("Momentum = %f" % (mm),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.40), textcoords='figure fraction')
pylab.annotate("Training size = %d" % (num_learn),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.35), textcoords='figure fraction')
pylab.annotate("Final accuracy = %f%%" % (val_accs[-1]),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.3), textcoords='figure fraction')
pylab.savefig('lasagne_conv_out_job%d_val_acc_vs_epoch_%s.pdf' % (tstamp, repo))
pylab.close()

fig = pylab.figure(figsize=(5, 5))
pylab.title('Validation loss vs epoch')
pylab.plot(epochs, val_loss)
pylab.xlabel('epoch')
pylab.ylabel('validation loss')
pylab.annotate(repo,
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.6), textcoords='figure fraction')
pylab.annotate("Learning rate = %f" % (lr),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.55), textcoords='figure fraction')
pylab.annotate("Momentum = %f" % (mm),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.50), textcoords='figure fraction')
pylab.annotate("Training size = %d" % (num_learn),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.45), textcoords='figure fraction')
pylab.savefig('lasagne_conv_out_job%d_val_loss_vs_epoch_%s.pdf' % (tstamp, repo))
pylab.close()
