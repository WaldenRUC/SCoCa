import argparse
import random
import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from BertContrasPretrain import BertContrastive
from transformers import BertTokenizer, BertModel
from file_preprocess_dataset import ContrasDataset
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=128,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=128,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--temperature",
                    default=0.1,
                    type=float,
                    help="The temperature for CL.")
parser.add_argument("--task",
                    default="aol",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=4,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--data_dir",
                    default="./model/",
                    type=str,
                    help="The path of the data.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--output_path",
                    default="./output/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to score file.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--bert_model_path",
                    default="../BERT/BertModel/",
                    type=str,
                    help="The path to BERT model.")
parser.add_argument("--pretrain_model_path",
                    default="../CLModel/BertContrastive.aol",
                    type=str)
parser.add_argument("--device_id",
                    default="0",
                    type=str,
                    help="GPU device id.")
parser.add_argument("--is_multiGPU",
                    default="1",
                    type=str,
                    help="是否用全部卡训练（默认单卡）")
args = parser.parse_args()
#--------------------------#
if args.is_multiGPU == "1":    #占用所有的卡
    args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
    args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
else:                          #单卡，用**device_id**指定
    args.batch_size = args.per_gpu_batch_size
    args.test_batch_size = args.per_gpu_test_batch_size
#--------------------------#
args.save_path += BertContrastive.__name__ + "." +  args.task + "." + str(args.epochs) + "." + str(int(args.temperature * 100)) + "." + str(args.per_gpu_batch_size)
args.loss_path = args.log_path + BertContrastive.__name__ + "." + args.task + "." + "train_cl_loss" + ".log"
args.log_path += BertContrastive.__name__ + "." + args.task + ".log"
args.score_file_path = args.output_path + BertContrastive.__name__ + "." + args.task + "." +  args.score_file_path

logger = open(args.log_path, "a")
loss_logger = open(args.loss_path, "a")
device = torch.device(f"cuda:{args.device_id}")
print(args, flush=True)
print(torch.cuda.current_device(), flush=True)
if args.is_multiGPU == "1":
    print("WARNING: use multiple gpus", flush=True)
else:
    print("WARNING: use single gpu", flush=True)
logger.write("\n")
seq_max_len = 128

if args.task == "aol":
    train_data = args.data_dir + "data/aol/train.pos.txt"
    test_data = args.data_dir + "data/aol/dev.pos.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 3
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
elif args.task == "tiangong":
    train_data = args.data_dir + "data/tiangong/train.pos.txt"
    test_data = args.data_dir + "data/tiangong/dev.pos.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 4
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
else:
    assert False

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_model():
    # load model
    if args.task == "aol":
        bert_model = BertModel.from_pretrained(args.bert_model_path)
    elif args.task == "tiangong":
        bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    model_state_dict = torch.load(args.pretrain_model_path, map_location="cuda:%s"%(args.device_id))
    bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)
    model = BertContrastive(bert_model, args=args, temperature=args.temperature)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params, flush=True)
    model = model.to(device)
    if args.is_multiGPU == "1":
        model = torch.nn.DataParallel(model)        #! 缺少device_ids参数，暂未修改
    fit(model, train_data, test_data)

def train_step(model, train_data, loss_func):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    contras_loss, acc = model.forward(train_data)
    return contras_loss, acc

def fit(model, X_train, X_test):
    train_dataset = ContrasDataset(X_train, seq_max_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(len(train_dataset) // args.batch_size), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = 1e4

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs, flush=True)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        loss_logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        # epoch_iterator = tqdm(train_dataloader, ncols=120)
        for i, training_data in enumerate(train_dataloader):
            loss, acc = train_step(model, training_data, bce_loss)
            loss = loss.mean()
            acc = acc.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            # epoch_iterator.set_postfix(lr=args.learning_rate, cont_loss=loss.item(), acc=acc.item())

            if i > 0 and i % 100 == 0:
                print("Step " + str(i) + ": " + str(loss.item()), flush=True)
                loss_logger.write("Step " + str(i) + ": " + str(loss.item()) + "\n")
                loss_logger.flush()

            if i > 0 and i % (one_epoch_step // 5) == 0:
            # if i > 0 and i % 10 == 0:
                best_result = evaluate(model, X_test, best_result)
                model.train()

            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        print("Average loss:{:.6f} ".format(avg_loss / cnt), flush=True)
        best_result = evaluate(model, X_test, best_result)
    logger.close()
    loss_logger.close()

def evaluate(model, X_test, best_result, is_test=False):
    y_test_loss, y_test_acc = predict(model, X_test)
    result = np.mean(y_test_loss)
    y_test_acc = np.mean(y_test_acc)

    if not is_test and result < best_result:
        best_result = result
        print("Best Result: Loss: %.4f Acc: %.4f" % (best_result, y_test_acc), flush=True)
        logger.write("Best Result: Loss: %.4f Acc: %.4f\n" % (best_result, y_test_acc))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    
    return best_result

def predict(model, X_test):
    model.eval()
    test_loss = []
    test_dataset = ContrasDataset(X_test, 128, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_test_loss = []
    y_test_acc = []
    with torch.no_grad():
        # epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)   #* leave=False: 执行后，清除进度条
        for i, test_data in enumerate(test_dataloader):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            test_loss, test_acc = model.forward(test_data)
            test_loss = test_loss.mean()
            test_acc = test_acc.mean()
            y_test_loss.append(test_loss.item())
            y_test_acc.append(test_acc.item())
    y_test_loss = np.asarray(y_test_loss)
    y_test_acc = np.asarray(y_test_acc)
    return y_test_loss, y_test_acc

if __name__ == '__main__':
    set_seed()
    if args.is_training:
        train_model()
