#  Code to generate training loss and evaluation metrics plots

# Plot 1 - Distillation Loss for each experiment
# Plot 2 - Finetuning Train and Evaluation Loss for each experiment on ChestXray14
# Plot 3 - Finetuning Evaluation mAUC for each experiment on ChestXray14
# Plot 4 - Finetuning Train and Evaluation Loss for each experiment on CheXpert
# Plot 5 - Finetuning Evaluation mAUC for each experiment on CheXpert

import matplotlib.pyplot as plt

# Define file paths
distill_files = {
    'Small->Tiny': '../experiment_logs/distill_small_tiny_100epochs_4gpus.txt',
    'Base->Tiny': '../experiment_logs/distill_base_tiny_100epochs_4gpus.txt',
}
    
finetune_chestxray14_files = {
    'Small->Tiny': '../experiment_logs/finetune_chestxray14_small_tiny_100epochs_4gpus.txt',
    'Base->Tiny': '../experiment_logs/finetune_chestxray14_base_tiny_100epochs_4gpus.txt',
}

finetune_chexpert_files = {
    'Small->Tiny': '../experiment_logs/finetune_chexpert_small_tiny_100epochs_4gpus.txt',
    'Base->Tiny': '../experiment_logs/finetune_chexpert_base_tiny_100epochs_4gpus.txt',
}

output_dir = '.'

distill_data = {}
# Read files
for key, distill_file in distill_files.items():
    with open(distill_file, 'r') as file:
        distill_log = file.readlines()
        distill_log = [eval(line) for line in distill_log]
        distill_data[key] = distill_log

finetune_chestxray14_data = {}
for key, finetune_file in finetune_chestxray14_files.items():
    with open(finetune_file, 'r') as file:
        finetune_log = file.readlines()
        finetune_log = [eval(line) for line in finetune_log]
        finetune_chestxray14_data[key] = finetune_log

finetune_chexpert_data = {}
for key, finetune_file in finetune_chexpert_files.items():
    with open(finetune_file, 'r') as file:
        finetune_log = file.readlines()
        finetune_log = [eval(line) for line in finetune_log]
        finetune_chexpert_data[key] = finetune_log

# Distillation loss plot
for key, distill_log in distill_data.items():
    train_loss = []
    epoch = []
    for line in distill_log:
        train_loss.append(line['train_loss'])
        epoch.append(line['epoch'])
    plt.plot(epoch, train_loss, label=key)
plt.title('Distillation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(output_dir + '/knowledge_distillation_loss.png')
plt.clf()

# CHESTXRAY14 PLOTS

# Finetuning loss plot for chestxray14
for key, finetune_log in finetune_chestxray14_data.items():
    train_loss = []
    test_loss = []
    epoch = []
    for line in finetune_log:
        train_loss.append(line['train_loss'])
        test_loss.append(line['test_loss'])
        epoch.append(line['epoch'])
    plt.plot(epoch, train_loss, label=key + ' (train)')
    plt.plot(epoch, test_loss, label=key + ' (eval)')
plt.title('Finetuning Train and Evaluation Loss on ChestXray14')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(output_dir + '/finetune_loss_chestxray14.png')
plt.clf()

# Finetuning evaluation metrics plot for chestxray14
for key, finetune_log in finetune_chestxray14_data.items():
    eval_acc = []
    epoch = []
    for line in finetune_log:
        eval_acc.append(line['test_auc_avg'])
        epoch.append(line['epoch'])
    plt.plot(epoch, eval_acc, label=key)
plt.title('Evaluation mAUC on ChestXray14')
plt.xlabel('Epoch')
plt.ylabel('mAUC')
plt.legend()
plt.savefig(output_dir + '/finetune_auc_chestxray14.png')
plt.clf()

# CHEXPERT PLOTS

# Finetuning loss plot for chestxray14
for key, finetune_log in finetune_chexpert_data.items():
    train_loss = []
    test_loss = []
    epoch = []
    for line in finetune_log:
        train_loss.append(line['train_loss'])
        test_loss.append(line['test_loss'])
        epoch.append(line['epoch'])
    plt.plot(epoch, train_loss, label=key + ' (train)')
    plt.plot(epoch, test_loss, label=key + ' (eval)')
plt.title('Finetuning Train and Evaluation Loss on CheXpert')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(output_dir + '/finetune_loss_chexpert.png')
plt.clf()

# Finetuning evaluation metrics plot for chexpert
for key, finetune_log in finetune_chexpert_data.items():
    eval_acc = []
    epoch = []
    for line in finetune_log:
        eval_acc.append(line['test_auc_avg'])
        epoch.append(line['epoch'])
    plt.plot(epoch, eval_acc, label=key)
plt.title('Evaluation mAUC on CheXpert')
plt.xlabel('Epoch')
plt.ylabel('mAUC')
plt.legend()
plt.savefig(output_dir + '/finetune_auc_chexpert.png')
plt.clf()