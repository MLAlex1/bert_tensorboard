from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from transformers import BertForSequenceClassification,BertConfig,BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

from utils_bert_mps import *

def train_epoch(  model : BertForSequenceClassification,
                  data_loader : DataLoader,
                  optimizer : AdamW,
                  scheduler : get_linear_schedule_with_warmup,
                  n_examples : int,
                  out_tensorboard : bool = True,
                  out_every : int = 30,
                  step_eval : int = None,
                  test_data_loader : DataLoader= None,
                  len_test_dataset : int = None
                ) -> Tuple[float, float]:
    """
    out_every : every how many steps add gradients and ratios figures & train
     loss to the tensorboard
    out_tensorboard : write to tensorboard or not
    step_eval : every how many step evaluate the model on test data. If None
    is passed then we will evaluate only at the end of the epoch.
    """
    print(f"Overall number of steps for training : {len(data_loader)*EPOCHS}")
    print(f"Tensorboard will save {(len(data_loader) * EPOCHS) // out_every} "
          f"figures")
    global counter_train
    SKIP_PROB = 0
    model = model.train()
    losses = []
    correct_predictions = 0

    tot_batches = len(data_loader) * EPOCHS
    steps_out_stats = list(np.arange(0, tot_batches, out_every))

    running_losses = []
    for d in tqdm(data_loader, desc="Train batch"):

        outputs = model(**d)
        preds = outputs.logits.argmax(1)
        loss = outputs.loss
        correct_predictions += torch.sum(preds == d['labels'])
        running_losses.append(loss.item())
        losses.append(loss.item())
        loss.backward()

        if counter_train in steps_out_stats and out_tensorboard:
            curr_params = copy.deepcopy(optimizer.param_groups[0]['params'])

            print("writing gradients and ratios..")
            # write gradients to tensorboard
            myfig=plot_grad_flow(model.named_parameters(), skip_prob=SKIP_PROB)
            writer.add_figure("gradients", myfig, global_step=counter_train,
                              close=True, walltime=None)

            named_params = copy_model_params(model.named_parameters())

            writer.add_scalar('loss/train', np.mean(running_losses),
                              counter_train)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            next_params = copy.deepcopy(optimizer.param_groups[0]['params'])
            ratios = compute_ratios(curr_params, next_params, named_params)
            optimizer.zero_grad()

            fig_ratio = plot_ratios(ratios, skip_prob=SKIP_PROB)
            writer.add_figure("gradient ratios", fig_ratio,
                global_step=counter_train, close=True, walltime=None)

        else:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        writer.add_scalar('learning_rate',scheduler.get_lr()[0],counter_train)

        if step_eval != None and counter_train % step_eval == 0 and \
                counter_train > 1:
            print("evaluating the model..")
            val_acc, val_loss = eval_model(model, test_dataset_loader,
                                           len_test_dataset)
            writer.add_scalar('loss/test', val_loss, counter_train)
            writer.add_scalar('accuracy/test', val_acc, counter_train)
            model = model.train()

        counter_train += 1
    return correct_predictions.cpu().numpy() / n_examples, np.mean(losses)

def eval_model(model : BertForSequenceClassification,
               data_loader : DataLoader,
               n_examples : int
              ) -> Tuple[float, float] :

    model = model.eval()
    losses = []
    correct_predictions = 0
    counter = 0
    with torch.no_grad():
        for d in tqdm(data_loader, "Eval batch"):
            counter += 1
            outputs = model(**d)
            preds = outputs.logits.argmax(1)
            loss = outputs.loss
            correct_predictions += torch.sum(preds == d['labels'])
            losses.append(loss.item())

    if torch.has_mps: #mps doesn't support double precision yet
        print ("evaluating with single precision")
        return correct_predictions.cpu().numpy() / n_examples, \
        np.mean(losses)
    else:
        print("evaluating with double precision")
        return correct_predictions.double().cpu().numpy() / n_examples, \
        np.mean(losses)

if __name__ == "__main__":
    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']

    newsgroups_train=fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test=fetch_20newsgroups(subset='test', categories=categories)

    print(list(newsgroups_train.target_names))

    X_train = pd.DataFrame(newsgroups_train['data'])
    y_train = pd.Series(newsgroups_train['target'])

    X_test = pd.DataFrame(newsgroups_test['data'])
    y_test = pd.Series(newsgroups_test['target'])

    print(f"Median length sentences "
          f"{X_train[0].apply(lambda x: len(x.split())).median()}")

    BATCH_SIZE = 16

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('mps' if torch.has_mps else 'cpu')
    print(f"Device : {device}")

    max_length = 256
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = len(y_train.unique())
    config.max_position_embeddings = max_length
    model = BertForSequenceClassification(config)
    model = model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_encodings = tokenizer(X_train[0].tolist(), truncation=True,
        padding=True, max_length=max_length)
    test_encodings= tokenizer(X_test[0].tolist(), truncation=True,
        padding=True, max_length=max_length)
    train_dataset = BertDataset(train_encodings, y_train)
    test_dataset = BertDataset(test_encodings, y_test)
    train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    writer = SummaryWriter('tensorboard/runs/bert_experiment_1')

    EPOCHS = 5
    # optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    total_steps = len(train_dataset_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps= 0.1 * total_steps,
      num_training_steps=total_steps
    )

    best_accuracy = 0
    counter_train = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(model, train_dataset_loader,
                                            optimizer, scheduler, len(train_dataset),
                                            step_eval=None, test_data_loader=test_dataset_loader,
                                            len_test_dataset = len(test_dataset), out_tensorboard=True)
        writer.add_scalar('accuracy/train', train_acc, counter_train)
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(model, test_dataset_loader,  len(test_dataset))
        writer.add_scalar('loss/test', val_loss, counter_train)
        writer.add_scalar('accuracy/test', val_acc, counter_train)
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
