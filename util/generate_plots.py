#  Code to generate training loss and evaluation metrics plots

import matplotlib.pyplot as plt

# Define file paths
distill_files = {
    'Small->Tiny': '../experiment_logs/distill_small_tiny_100epochs_4gpus.txt',
    'Base->Tiny': '../experiment_logs/distill_base_tiny_100epochs_4gpus.txt',
}
    
finetune_files = {
    'Small->Tiny': '../experiment_logs/finetune_chestxray14_small_tiny_100epochs_4gpus.txt',
    'Base->Tiny': '../experiment_logs/finetune_chestxray14_base_tiny_100epochs_4gpus.txt',
}

output_dir = '.'

distill_data = {}
# Read files
for key, distill_file in distill_files.items():
    with open(distill_file, 'r') as file:
        distill_log = file.readlines()
        distill_log = [eval(line) for line in distill_log]
        distill_data[key] = distill_log

finetune_data = {}
for key, finetune_file in finetune_files.items():
    with open(finetune_file, 'r') as file:
        finetune_log = file.readlines()
        finetune_log = [eval(line) for line in finetune_log]
        finetune_data[key] = finetune_log

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
plt.savefig(output_dir + '/distill_loss.png')
plt.clf()

# Finetuning loss plot
for key, finetune_log in finetune_data.items():
    train_loss = []
    epoch = []
    for line in finetune_log:
        train_loss.append(line['train_loss'])
        epoch.append(line['epoch'])
    plt.plot(epoch, train_loss, label=key)
plt.title('Fine-tune Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(output_dir + '/finetune_loss.png')
plt.clf()

# Finetuning evaluation metrics plot
for key, finetune_log in finetune_data.items():
    eval_acc = []
    epoch = []
    for line in finetune_log:
        eval_acc.append(line['test_auc_avg'])
        epoch.append(line['epoch'])
    plt.plot(epoch, eval_acc, label=key)
plt.title('Mean AUC on Valid Set')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.savefig(output_dir + '/finetune_eval.png')
plt.clf()
