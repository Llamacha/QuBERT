import pandas as pd
from simpletransformers.ner import NERModel,NERArgs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

train_data = pd.read_csv("../NER_train_bpeguided_seg.csv")
test_data = pd.read_csv("../NER_test_bpeguided_seg.csv")
label = train_data["labels"].unique().tolist()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

def get_accuracy_and_confusion (labels, preds):
    pred_list = list()
    label_list = list()

    for pred in preds:
        pred_list.append(pred[0])
    for label in labels:
        label_list.append(label[0])

    labels = label_list
    preds = pred_list
    label_set = list(set(labels))
    label_set.sort()

    print(label_set)
    matrix = confusion_matrix(labels, preds, labels=label_set)
    print(matrix)
    labels = mlb.fit_transform(labels)
    preds = mlb.transform(preds)

    return accuracy_score(labels, preds)
args = NERArgs()
args.num_train_epochs = 10
args.learning_rate = 1e-4
args.overwrite_output_dir =True
args.train_batch_size = 32
args.eval_batch_size = 32
args.wandb_project = 'quechuaNER'
args.manual_seed = 42
args.evaluate_during_training = False
args.save_model_every_epoch = False
args.save_steps = -1
model = NERModel('roberta', "../BPEGquechuaBERT" ,labels=label, args =args,use_cuda=True)

model.train_model(train_data,eval_data = test_data, acc=get_accuracy_and_confusion)

result, model_outputs, preds_list = model.eval_model(test_data, acc=get_accuracy_and_confusion)

print(result)