import argparse

from main import smoke_test, test, train
from util import load_config

base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument("--config", default="config.yaml", type=str)
base_args, _ = base_parser.parse_known_args()
app_config = load_config(base_args.config)
defaults = app_config.get("defaults", {})

parser = argparse.ArgumentParser(description="Model Controller")
parser.add_argument("--config", default=base_args.config, type=str)
parser.add_argument("--cuda_id", default=defaults.get("cuda_id", "0"), type=str)
parser.add_argument("--dataset", default=defaults.get("dataset", "WebNLG"), type=str)
parser.add_argument("--rounds", default=defaults.get("rounds", 4), type=int)
parser.add_argument("--train", default=defaults.get("train", "train"), type=str)

parser.add_argument("--batch_size", default=defaults.get("batch_size", 6), type=int)
parser.add_argument("--test_batch_size", default=defaults.get("test_batch_size", 6), type=int)
parser.add_argument("--learning_rate", default=defaults.get("learning_rate", 3e-5), type=float)
parser.add_argument("--num_train_epochs", "--epochs", default=defaults.get("num_train_epochs", 50), type=int)
parser.add_argument("--fix_bert_embeddings", default=defaults.get("fix_bert_embeddings", False), type=bool)
parser.add_argument("--bert_vocab_path", default=defaults.get("bert_vocab_path", "./pretrained/bert-base-cased/vocab.txt"), type=str)
parser.add_argument("--bert_config_path", default=defaults.get("bert_config_path", "./pretrained/bert-base-cased/config.json"), type=str)
parser.add_argument("--bert_model_path", default=defaults.get("bert_model_path", "./pretrained/bert-base-cased/pytorch_model.bin"), type=str)
parser.add_argument("--max_len", default=defaults.get("max_len", 100), type=int)
parser.add_argument("--warmup", default=defaults.get("warmup", 0.0), type=float)
parser.add_argument("--weight_decay", default=defaults.get("weight_decay", 0.0), type=float)
parser.add_argument("--max_grad_norm", default=defaults.get("max_grad_norm", 1.0), type=float)
parser.add_argument("--min_num", default=defaults.get("min_num", 1e-7), type=float)
parser.add_argument("--base_path", default=defaults.get("base_path", "./dataset"), type=str)
parser.add_argument("--output_path", default=defaults.get("output_path", "./ckpt"), type=str)
parser.add_argument("--save_interval", default=defaults.get("save_interval", 1), type=int)
parser.add_argument("--seed", default=defaults.get("seed", 0), type=int)
parser.add_argument("--load_model", action="store_true")
parser.add_argument("--smoke_test", action="store_true")

args = parser.parse_args()

if args.smoke_test:
    smoke_test(args)
elif args.train == "train":
    train(args)
else:
    test(args)
