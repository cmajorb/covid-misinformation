import sys
import ast
import matplotlib.pyplot as plt
import re
pattern = re.compile("{.*}")



hyper_parameters = {
  "with_training" : [True],
  "model_name" : ["dmis-lab/biobert-base-cased-v1.2"],
  "data_set" : ["nela","coaid"],
  "per_gpu_batch_size": [4, 8],
  "learning_rate": [2e-5, 3e-5, 5e-5],
  "num_epochs": [4],
  "train_split": [0.1,0.66]
}

configurations = []
for with_training in hyper_parameters["with_training"]:
    for model_name in hyper_parameters["model_name"]:
        for data_set in hyper_parameters["data_set"]:
            for per_gpu_batch_size in hyper_parameters["per_gpu_batch_size"]:
                for learning_rate in hyper_parameters["learning_rate"]:
                    for num_epochs in hyper_parameters["num_epochs"]:
                        for train_split in hyper_parameters["train_split"]:
                            configurations.append({
                                "with_training": with_training,
                                "model_name": model_name,
                                "data_set": data_set,
                                "per_gpu_batch_size": per_gpu_batch_size,
                                "learning_rate": learning_rate,
                                "num_epochs": num_epochs,
                                "train_split": train_split
                            })


#slurm_id = sys.argv[1]
slurm_id = "14891821"

fig, ax = plt.subplots(nrows=8, ncols=3,figsize=(10,15))
axes = ax.flat

'''
for i, config in enumerate(configurations):
    for index, line in enumerate(open("slurm-" + slurm_id + '_'+ str(i) +'.out')):
        for match in re.finditer(pattern, line):
            dict = ast.literal_eval(match.group())
            if 'eval_loss' in dict:
                print(i)
                print(dict['eval_accuracy'])
                print(dict['eval_loss'])
                print("----")

'''
for i, config in enumerate(configurations):
    epoch = []
    accuracy = []
    loss = []
    for index, line in enumerate(open("slurm-" + slurm_id + '_'+ str(i) +'.out')):
        for match in re.finditer(pattern, line):
            dict = ast.literal_eval(match.group())
            if 'eval_loss' in dict:
                epoch.append(dict['epoch'])
                accuracy.append(dict['eval_accuracy'])
                loss.append(dict['eval_loss'])
    title = str(i) + " (b: " + str(config["per_gpu_batch_size"]) + ", lr: " + str(config["learning_rate"]) + "s: " + str(config["train_split"]) + "ds" + config["data_set"]+")"
    color = 'tab:red'
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Accuracy', color=color)
    axes[i].plot(epoch, accuracy, label = "accuracy", color=color)

    color = 'tab:blue'
    ax2 = axes[i].twinx()
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(epoch, loss, label = "loss", color=color)
    axes[i].set_title(title)

fig.tight_layout()
plt.show()
plt.savefig(slurm_id + '.png')
