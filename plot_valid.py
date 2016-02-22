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
l2_reg_match = re.compile(r"^ L2 regularization penalty scale:")
batch_size_match = re.compile(r"^ Batch size:")
adagrad_match = re.compile(r"Use AdaGrad")

epochs = []
val_accs = []
val_loss = []
run_times = []
learning_rate_schedule = ''

learning_rate = -1.0
momentum = -1.0
l2_pen = -1.0
batch_sz = -1

# we just want one copy of the values we mine from the log file in case a
# prediction pass is run after the training and the executor was lazy about
# matching parameters
repo_found = False
learning_rate_found = False
momentum_found = False
datasize_found = False
l2_pen_found = False
batch_sz_found = False
learning_rate_schedule_found = False

tstamp = int(re.findall(r"[0-9]+", logname)[0])
print("Log timestamp =", tstamp)

with open(logname, 'r') as f:
    lines = f.readlines()
    for line in lines:
        gt_m = re.search(git_repo_match, line)
        learning_rate_m = re.search(learning_rate_match, line)
        momentum_m = re.search(momentum_match, line)
        datasize_m = re.search(datasize_match, line)
        epoch_m = re.search(epoch_match, line)
        val_acc_m = re.search(val_acc_match, line)
        val_loss_m = re.search(val_loss_match, line)
        l2_pen_m = re.search(l2_reg_match, line)
        batch_sz_m = re.search(batch_size_match, line)
        adagrad_m = re.search(adagrad_match, line)
        if gt_m and not repo_found:
            repo_found = True
            repo = line.split()[-1]
            print("repo hash =", repo)
        if learning_rate_m and not learning_rate_found:
            learning_rate_found = True
            learning_rate = float(line.split(':')[-1])
            print("learning rate =", learning_rate)
        if momentum_m and not momentum_found:
            momentum_found = True
            momentum = float(line.split(':')[-1])
            print("momentum = ", momentum)
        if datasize_m and not datasize_found:
            datasize_found = True
            nums_m = re.findall(r"[0-9]+", line)
            num_learn = int(nums_m[0])
            print(num_learn)
            img_w = int(nums_m[2])
            img_h = int(nums_m[3])
            print("img w, h = ", img_w, ",", img_h)
        if l2_pen_m and not l2_pen_found:
            l2_pen_found = True
            l2_pen = float(line.split(':')[-1])
            print("l2 reg pen = ", l2_pen)
        if batch_sz_m and not batch_sz_found:
            batch_sz_found = True
            batch_sz = int(line.split(':')[-1])
            print("batch size = ", batch_sz)
        if adagrad_m and not learning_rate_schedule_found:
            learning_rate_schedule_found = True
            learning_rate_schedule = "AdaGrad"
            print("learning schedule = ", learning_rate_schedule)
        if epoch_m:
            nums_e = re.findall(r"[0-9]+", line)
            epochs.append(int(nums_e[0]))
            runt = float(str(nums_e[2] + '.' + nums_e[3]))
            run_times.append(runt)
        if val_acc_m:
            val_accs.append(float(line.split()[-2]))
        if val_loss_m:
            val_loss.append(float(line.split()[-1]))

epochs = np.asarray(epochs)
val_accs = np.asarray(val_accs)
val_loss = np.asarray(val_loss)
# change run times from seconds to minutes
run_times = np.cumsum(np.asarray(run_times)) / 60.0

fig = pylab.figure(figsize=(6, 6))
pylab.title('Validation accuracy vs epoch')
pylab.plot(epochs, val_accs)
pylab.xlabel('epoch')
pylab.ylabel('validation accuracy (%)')
pylab.annotate(repo,
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.5), textcoords='figure fraction')
pylab.annotate("Learning rate = %f" % (learning_rate),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.45), textcoords='figure fraction')
pylab.annotate("Learning schedule = %s" % (learning_rate_schedule),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.4), textcoords='figure fraction')
pylab.annotate("Momentum = %f" % (momentum),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.35), textcoords='figure fraction')
pylab.annotate("L2 penalty = %f" % (l2_pen),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.3), textcoords='figure fraction')
pylab.annotate("Training size = %d" % (num_learn),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.25), textcoords='figure fraction')
pylab.annotate("Batch size = %d" % (batch_sz),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.2), textcoords='figure fraction')
pylab.annotate("Final accuracy = %f%%" % (val_accs[-1]),
               xy=(epochs[len(epochs) // 2], val_accs[len(val_accs) // 2]), 
               xytext=(0.35, 0.15), textcoords='figure fraction')
pylab.savefig('lasagne_conv_out_job%d_val_acc_vs_epoch_%s.pdf' % (tstamp, repo))
pylab.close()

fig = pylab.figure(figsize=(6, 6))
pylab.title('Validation loss vs epoch')
pylab.plot(epochs, val_loss)
pylab.xlabel('epoch')
pylab.ylabel('validation loss')
pylab.annotate(repo,
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.6), textcoords='figure fraction')
pylab.annotate("Learning rate = %f" % (learning_rate),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.55), textcoords='figure fraction')
pylab.annotate("Learning schedule = %s" % (learning_rate_schedule),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.5), textcoords='figure fraction')
pylab.annotate("Momentum = %f" % (momentum),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.45), textcoords='figure fraction')
pylab.annotate("L2 penalty = %f" % (l2_pen),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.4), textcoords='figure fraction')
pylab.annotate("Training size = %d" % (num_learn),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.35), textcoords='figure fraction')
pylab.annotate("Batch size = %d" % (batch_sz),
               xy=(epochs[len(epochs) // 2], val_loss[len(val_loss) // 2]), 
               xytext=(0.4, 0.3), textcoords='figure fraction')
pylab.savefig('lasagne_conv_out_job%d_val_loss_vs_epoch_%s.pdf' % (tstamp, repo))
pylab.close()

fig = pylab.figure(figsize=(6, 6))
pylab.title('Cumulative run time vs epoch')
pylab.plot(epochs, run_times)
pylab.xlabel('epoch')
pylab.ylabel('run time (m)')
pylab.annotate(repo,
               xy=(epochs[len(epochs) // 2], run_times[len(run_times) // 2]), 
               xytext=(0.25, 0.8), textcoords='figure fraction')
pylab.annotate("Learning rate = %f" % (learning_rate),
               xy=(epochs[len(epochs) // 2], run_times[len(run_times) // 2]), 
               xytext=(0.25, 0.75), textcoords='figure fraction')
pylab.annotate("Momentum = %f" % (momentum),
               xy=(epochs[len(epochs) // 2], run_times[len(run_times) // 2]), 
               xytext=(0.25, 0.70), textcoords='figure fraction')
pylab.annotate("Training size = %d" % (num_learn),
               xy=(epochs[len(epochs) // 2], run_times[len(run_times) // 2]), 
               xytext=(0.35, 0.25), textcoords='figure fraction')
pylab.annotate("Total time = %f m" % (run_times[-1]),
               xy=(epochs[len(epochs) // 2], run_times[len(run_times) // 2]), 
               xytext=(0.35, 0.2), textcoords='figure fraction')
pylab.savefig('lasagne_conv_out_job%d_run_times_vs_epoch_%s.pdf' % (tstamp, repo))
pylab.close()
