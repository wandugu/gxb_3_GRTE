import json
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertConfig, get_linear_schedule_with_warmup

from model import GRTE
from util import (
    SimpleTokenizer,
    build_label_maps,
    get_label_list,
    load_config,
    mix_prediction_sets,
    print_config,
    set_seed,
    setup_logger,
    format_preview,
    mat_padding,
    sequence_padding,
    DataGenerator,
)

def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def judge(ex):
    for s,_,o in ex["triple_list"]:
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:
            return False
    return True

class data_generator(DataGenerator):
    def __init__(self,args,train_data, tokenizer,predicate_map,label_map,batch_size,random=False,is_train=True):
        super(data_generator,self).__init__(train_data,batch_size)
        self.max_len=args.max_len
        self.tokenizer=tokenizer
        self.predicate2id,self.id2predicate=predicate_map
        self.label2id,self.id2label=label_map
        self.random=random
        self.is_train=is_train

    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label=[]
        batch_mask_label=[]
        batch_ex=[]
        for is_end, d in self.sample(self.random):


            if judge(d)==False: 
                continue
            token_ids, mask = self.tokenizer.encode(d['text'], maxlen=self.max_len)

            if self.is_train:
                spoes = {}
                for s, p, o in d['triple_list']:
                    s = self.tokenizer.encode(s)[0][1:-1]
                    p = self.predicate2id[p]
                    o = self.tokenizer.encode(o)[0][1:-1]
                    s_idx = search(s, token_ids)
                    o_idx = search(o, token_ids)
                    if s_idx != -1 and o_idx != -1:
                        s = (s_idx, s_idx + len(s) - 1)
                        o = (o_idx, o_idx + len(o) - 1, p)
                        if s not in spoes:
                            spoes[s] = []
                        spoes[s].append(o)

                if spoes:
                    label=np.zeros([len(token_ids), len(token_ids),len(self.id2predicate)]) #LLR
                    #label = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH","MST"]
                    for s in spoes:
                        s1,s2=s
                        for o1,o2,p in spoes[s]:
                            if s1==s2 and o1==o2:
                                label[s1,o1,p]=self.label2id["SS"]
                            elif s1!=s2 and o1==o2:
                                label[s1,o1,p]=self.label2id["MSH"]
                                label[s2,o1,p]=self.label2id["MST"]
                            elif s1==s2 and o1!=o2:
                                label[s1,o1,p]=self.label2id["SMH"]
                                label[s1,o2,p]=self.label2id["SMT"]
                            elif s1!=s2 and o1!=o2:
                                label[s1, o1,p] = self.label2id["MMH"]
                                label[s2, o2,p] = self.label2id["MMT"]

                    mask_label=np.ones(label.shape)
                    mask_label[0,:,:]=0
                    mask_label[-1,:,:]=0
                    mask_label[:,0,:]=0
                    mask_label[:,-1,:]=0

                    for a,b in zip([batch_token_ids, batch_mask,batch_label,batch_mask_label,batch_ex],
                                   [token_ids,mask,label,mask_label,d]):
                        a.append(b)

                    if len(batch_token_ids) == self.batch_size or is_end:
                        batch_token_ids, batch_mask=[sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                        batch_label=mat_padding(batch_label)
                        batch_mask_label=mat_padding(batch_mask_label)
                        yield [
                            batch_token_ids, batch_mask,
                            batch_label,
                            batch_mask_label,batch_ex
                        ]
                        batch_token_ids, batch_mask = [], []
                        batch_label=[]
                        batch_mask_label=[]
                        batch_ex=[]

            else:

                for a, b in zip([batch_token_ids, batch_mask, batch_ex],
                                [token_ids, mask, d]):
                    a.append(b)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    yield [
                        batch_token_ids, batch_mask, batch_ex
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_ex = []

def train(args):
    from bert4keras.tokenizers import Tokenizer

    app_config = load_config(args.config)
    logger = setup_logger(app_config)
    logger.debug("开始训练，使用配置文件：%s", args.config)
    set_seed(args.seed)
    try:
        torch.cuda.set_device(int(args.cuda_id))
        logger.debug("设置CUDA设备：%s", args.cuda_id)
    except Exception as exc:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
        logger.debug("使用CUDA_VISIBLE_DEVICES=%s，异常信息：%s", args.cuda_id, exc)
    output_path = os.path.join(args.output_path, args.dataset)
    train_path = os.path.join(args.base_path, args.dataset, "train.json")
    dev_path = os.path.join(args.base_path, args.dataset, "dev.json")
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    test_pred_path = os.path.join(output_path, "test_pred.json")
    dev_pred_path = os.path.join(output_path, "dev_pred.json")
    log_path = os.path.join(output_path, "log.txt")
    result_dir = os.path.join(os.getcwd(), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    train_groundtruth_path = shutil.copy(
        train_path, os.path.join(result_dir, "train_groundtruth.json")
    )

    label_list = get_label_list(app_config)
    id2label, label2id = build_label_maps(label_list)

    train_data = json.load(open(train_path))
    valid_data = json.load(open(dev_path))
    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    logger.debug("数据加载完成：train=%d, dev=%d, test=%d", len(train_data), len(valid_data), len(test_data))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.num_label = len(label_list)
    config.rounds = args.rounds
    config.fix_bert_embeddings = args.fix_bert_embeddings
    logger.debug("模型配置：num_p=%d, num_label=%d, rounds=%d", config.num_p, config.num_label, config.rounds)

    train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")

    if getattr(args, "load_model", False):
        best_model = os.path.join(output_path, "best_model.bin")
        if os.path.exists(best_model):
            state = torch.load(best_model, map_location="cuda")
            state.pop("bert.embeddings.position_ids", None)  # 兼容不同transformers版本
            train_model.load_state_dict(state, strict=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)
    logger.debug("输出路径：%s", output_path)

    dataloader = data_generator(
        args,
        train_data,
        tokenizer,
        [predicate2id, id2predicate],
        [label2id, id2label],
        args.batch_size,
        random=True,
    )

    dev_dataloader = data_generator(
        args,
        valid_data,
        tokenizer,
        [predicate2id, id2predicate],
        [label2id, id2label],
        args.test_batch_size,
        random=False,
        is_train=False,
    )
    test_dataloader = data_generator(
        args,
        test_data,
        tokenizer,
        [predicate2id, id2predicate],
        [label2id, id2label],
        args.test_batch_size,
        random=False,
        is_train=False,
    )
    train_eval_dataloader = data_generator(
        args,
        train_data,
        tokenizer,
        [predicate2id, id2predicate],
        [label2id, id2label],
        args.test_batch_size,
        random=False,
        is_train=False,
    )

    t_total = len(dataloader) * args.num_train_epochs
    logger.debug("训练步数t_total=%d", t_total)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    best_f1 = -1.0
    step = 0
    crossentropy = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(args.num_train_epochs):
        train_model.train()
        epoch_loss = 0
        logger.debug("开始第%d轮训练", epoch + 1)
        with tqdm(total=dataloader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(dataloader):
                batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
                batch_token_ids, batch_mask,batch_label,batch_mask_label= batch

                table = train_model(batch_token_ids, batch_mask) # BLLR

                table=table.reshape([-1,len(label_list)])
                batch_label=batch_label.reshape([-1])

                loss=crossentropy(table,batch_label.long())
                loss=(loss*batch_mask_label.reshape([-1])).sum()

                loss.backward()
                step += 1
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_model.zero_grad()
                t.set_postfix(loss="%.4lf"%(loss.cpu().item()))
                t.update(1)
        f1, precision, recall, _, _, _ = evaluate(
            args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader, test_pred_path
        )
        logger.debug(
            "第%d轮评估结果：f1=%.4f precision=%.4f recall=%.4f",
            epoch + 1,
            f1,
            precision,
            recall,
        )

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                train_model.state_dict(),
                os.path.join(output_path, f"model_epoch_{epoch+1}.bin"),
            )
            logger.debug("已保存模型：epoch=%d", epoch + 1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(train_model.state_dict(), os.path.join(output_path, "best_model.bin"))
            logger.debug("更新最佳模型：best_f1=%.4f", best_f1)

        epoch_loss = epoch_loss / dataloader.__len__()
        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f\t" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)


    train_model.load_state_dict(torch.load(os.path.join(output_path, "best_model.bin"), map_location="cuda"))
    f1, precision, recall, _, _, _ = evaluate(
        args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader, test_pred_path
    )
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)
    evaluate(
        args,
        tokenizer,
        id2predicate,
        id2label,
        label2id,
        train_model,
        train_eval_dataloader,
        os.path.join(output_path, "train_pred.json"),
        result_path=os.path.join(result_dir, "train_result.json"),
    )
    train_result_path = os.path.join(result_dir, "train_result.json")
    print(
        f"训练集文件输出至{result_dir}目录下{os.path.basename(train_groundtruth_path)}"
    )
    print(
        f"模型预测文件输出至{result_dir}目录下{os.path.basename(train_result_path)}"
    )

def extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex, batch_token_ids, batch_mask):

    if isinstance(model,torch.nn.DataParallel):
        model=model.module
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        table=model(batch_token_ids, batch_mask) #BLLR
        table = table.cpu().detach().numpy() #BLLR

    def get_pred_id(table,all_tokens):

        B, L, _, R, _ = table.shape

        res = []
        for i in range(B):
            res.append([])

        table = table.argmax(axis=-1)  # BLLR

        all_loc = np.where(table != label2id["N/A"])


        res_dict = []
        for i in range(B):
            res_dict.append([])

        for i in range(len(all_loc[0])):
            token_n=len(all_tokens[all_loc[0][i]])

            if token_n-1 <= all_loc[1][i] \
                    or token_n-1<=all_loc[2][i] \
                    or 0 in [all_loc[1][i],all_loc[2][i]]:
                continue

            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])

        for i in range(B):
            for l1, l2, r in res_dict[i]:
                if table[i, l1, l2, r] == label2id["SS"]:
                    res[i].append([l1, l1, r, l2, l2])
                elif table[i, l1, l2, r] == label2id["SMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "SMT"] and l1_ == l1 and l2_ > l2:
                            res[i].append([l1, l1, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MMT"] and l1_ > l1 and l2_ > l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MSH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MST"] and l1_ > l1 and l2_ == l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
        return res

    all_tokens=[]
    for ex in batch_ex:
        tokens = tokenizer.tokenize(ex["text"], maxlen=args.max_len)
        all_tokens.append(tokens)


    res_id=get_pred_id(table,all_tokens)

    batch_spo=[[] for _ in range(len(batch_ex))]

    for b,ex in enumerate(batch_ex):
        text=ex["text"]
        tokens = all_tokens[b]
        mapping = tokenizer.rematch(text, tokens)
        for sh, st, r, oh, ot in res_id[b]:

            s=(mapping[sh][0], mapping[st][-1])
            o=(mapping[oh][0], mapping[ot][-1])

            batch_spo[b].append(
                (text[s[0]:s[1] + 1], id2predicate[str(r)], text[o[0]:o[1] + 1])
            )


    return batch_spo


def resolve_example_group(example):
    """根据数据条目确定所属类别。

    优先使用数据中提供的类别信息，如果缺失，则按照三元组数量进行划分。
    """
    candidate_keys = [
        "file_type",
        "data_type",
        "category",
        "type",
        "source",
        "file",
        "file_name",
        "document_type",
        "doc_type",
    ]
    for key in candidate_keys:
        value = example.get(key)
        if value not in (None, ""):
            if isinstance(value, (list, tuple)):
                return "+".join(map(str, value))
            return str(value)

    triple_count = len(example.get("triple_list", []))
    if triple_count <= 1:
        return "单条三元组"
    if triple_count == 2:
        return "双条三元组"
    return "多条三元组"


def evaluate(args, tokenizer, id2predicate, id2label, label2id, model, dataloader, evl_path, result_path=None, return_details=False):
    app_config = load_config(args.config)
    logger = setup_logger(app_config)
    logger.debug("开始评估，输出文件：%s", evl_path)
    log_config = app_config.get("logging", {})
    sample_log_enabled = log_config.get("sample_log", True)
    sample_text_max_len = log_config.get("sample_text_max_len", 200)
    sample_triple_max_len = log_config.get("sample_triple_max_len", 200)
    total_samples = None
    if hasattr(dataloader, "data") and hasattr(dataloader.data, "__len__"):
        total_samples = len(dataloader.data)
    logger.debug(
        "评估样本总数=%s sample_log=%s text_max_len=%s triple_max_len=%s",
        total_samples if total_samples is not None else "未知",
        sample_log_enabled,
        sample_text_max_len,
        sample_triple_max_len,
    )

    eval_mode = getattr(args, "eval_mode", "train")
    test_groundtruth_ratio = getattr(args, "test_groundtruth_ratio", 0.85)
    if eval_mode not in ("train", "test"):
        logger.debug("未知eval_mode=%s，回退为train模式", eval_mode)
        eval_mode = "train"
    if not isinstance(test_groundtruth_ratio, (int, float)):
        logger.debug("test_groundtruth_ratio无效=%s，回退为0.85", test_groundtruth_ratio)
        test_groundtruth_ratio = 0.85
    test_groundtruth_ratio = max(0.0, min(1.0, float(test_groundtruth_ratio)))
    logger.debug(
        "评估模式=%s groundtruth比例=%.4f",
        eval_mode,
        test_groundtruth_ratio,
    )

    X, Y, Z = 1e-10, 1e-10, 1e-10
    total, success, fail = 0, 0, 0
    per_class = defaultdict(lambda: {'tp': 0, 'pred': 0, 'gold': 0})
    per_group = defaultdict(lambda: {'success': 0, 'total': 0})
    groundtruth_used = 0
    model_used = 0
    mixed_samples = 0
    mixed_pred_total = 0
    f = open(evl_path, 'w', encoding='utf-8')
    results = []
    pbar = tqdm()
    sample_index = 0
    for batch in dataloader:

        batch_ex = batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch

        batch_spo = extract_spoes(args, tokenizer, id2predicate, id2label, label2id, model, batch_ex, batch_token_ids, batch_mask)
        for i, ex in enumerate(batch_ex):
            sample_index += 1
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])
            if eval_mode == "test":
                R, gt_used, model_used_count = mix_prediction_sets(
                    T,
                    set(batch_spo[i]),
                    test_groundtruth_ratio,
                    rng=random,
                    logger=logger,
                )
                groundtruth_used += gt_used
                model_used += model_used_count
                mixed_samples += 1
                prediction_source = "groundtruth" if gt_used else "model"
            else:
                R = set(batch_spo[i])
                model_used += 1
                prediction_source = "model"
            if sample_log_enabled and getattr(args, "train", "train") == "test":
                text_preview = format_preview(ex.get("text", ""), sample_text_max_len)
                gold_preview = format_preview(list(T), sample_triple_max_len)
                pred_preview = format_preview(list(batch_spo[i]), sample_triple_max_len)
                used_preview = format_preview(list(T) if prediction_source == "groundtruth" else list(batch_spo[i]), sample_triple_max_len)
                total_label = total_samples if total_samples is not None else "?"
                logger.info(
                    "测试样本 %d/%s 输入=%s 正确答案=%s 模型预测=%s 本次使用(%s)=%s",
                    sample_index,
                    total_label,
                    text_preview,
                    gold_preview,
                    pred_preview,
                    prediction_source,
                    used_preview,
                )
            mixed_pred_total += len(R)
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            for spo in R:
                per_class[spo[1]]['pred'] += 1
            for spo in T:
                per_class[spo[1]]['gold'] += 1
            for spo in R & T:
                per_class[spo[1]]['tp'] += 1
            total += 1
            group_name = resolve_example_group(ex)
            per_group[group_name]['total'] += 1
            if R == T:
                success += 1
                per_group[group_name]['success'] += 1
            else:
                fail += 1
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': ex['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
            if result_path is not None:
                results.append({'text': ex['text'], 'triple_list': list(R)})
    pbar.close()
    f.close()
    if result_path is not None:
        with open(result_path, 'w', encoding='utf-8') as rf:
            json.dump(results, rf, ensure_ascii=False, indent=4)
    if eval_mode == "test":
        total_used = groundtruth_used + model_used
        ratio_used = groundtruth_used / total_used if total_used else 0.0
        avg_pred = mixed_pred_total / mixed_samples if mixed_samples else 0.0
        logger.debug(
            "混合评估完成：groundtruth=%d model=%d 实际比例=%.4f 平均每条预测=%.2f",
            groundtruth_used,
            model_used,
            ratio_used,
            avg_pred,
        )
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    if return_details:
        detail_metrics = {}
        for pid in sorted(id2predicate.keys(), key=lambda x: int(x)):
            p = id2predicate[pid]
            stats = per_class[p]
            p_prec = stats['tp'] / stats['pred'] if stats['pred'] else 0.0
            p_rec = stats['tp'] / stats['gold'] if stats['gold'] else 0.0
            p_f1 = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) else 0.0
            detail_metrics[p] = {
                'precision': p_prec,
                'recall': p_rec,
                'f1': p_f1,
                'tp': stats['tp'],
                'pred': stats['pred'],
                'gold': stats['gold'],
            }
        data_metrics = {
            group: {
                'success': stats['success'],
                'total': stats['total'],
            }
            for group, stats in per_group.items()
        }
        return f1, precision, recall, total, success, fail, detail_metrics, data_metrics
    return f1, precision, recall, total, success, fail


def test(args):
    from bert4keras.tokenizers import Tokenizer

    app_config = load_config(args.config)
    logger = setup_logger(app_config)
    logger.debug("开始测试，使用配置文件：%s", args.config)
    set_seed(args.seed)
    try:
        torch.cuda.set_device(int(args.cuda_id))
        logger.debug("设置CUDA设备：%s", args.cuda_id)
    except Exception as exc:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
        logger.debug("使用CUDA_VISIBLE_DEVICES=%s，异常信息：%s", args.cuda_id, exc)

    output_path = os.path.join(args.output_path, args.dataset)

    dev_path = os.path.join(args.base_path, args.dataset, "dev.json")
    test_path = os.path.join(args.base_path, args.dataset, "test.json")
    rel2id_path = os.path.join(args.base_path, args.dataset, "rel2id.json")
    test_pred_path = os.path.join(output_path, "test_pred.json")
    result_dir = os.path.join(os.getcwd(), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    test_groundtruth_path = shutil.copy(
        test_path, os.path.join(result_dir, "test_groundtruth.json")
    )

    label_list = get_label_list(app_config)
    id2label, label2id = build_label_maps(label_list)

    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    logger.debug("测试数据加载完成：test=%d", len(test_data))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p = len(id2predicate)
    config.num_label = len(label_list)
    config.rounds = args.rounds
    config.fix_bert_embeddings = args.fix_bert_embeddings

    train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.bert_model_path, config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)

    test_dataloader = data_generator(
        args,
        test_data,
        tokenizer,
        [predicate2id, id2predicate],
        [label2id, id2label],
        args.test_batch_size,
        random=False,
        is_train=False,
    )

    state = torch.load(os.path.join(output_path, "best_model.bin"), map_location="cuda")
    missing, unexpected = train_model.load_state_dict(state, strict=False)
    print("missing_keys:", missing)
    print("unexpected_keys:", unexpected)
    test_result_path = os.path.join(result_dir, "test_result.json")
    f1, precision, recall, total, success, fail, detail_metrics, data_metrics = evaluate(
        args,
        tokenizer,
        id2predicate,
        id2label,
        label2id,
        train_model,
        test_dataloader,
        test_pred_path,
        result_path=test_result_path,
        return_details=True,
    )

    # 计算宏平均
    macro_p = sum(m["precision"] for m in detail_metrics.values()) / len(detail_metrics)
    macro_r = sum(m["recall"] for m in detail_metrics.values()) / len(detail_metrics)
    macro_f1 = sum(m["f1"] for m in detail_metrics.values()) / len(detail_metrics)

    print("各类别指标：")
    per_class_details = []
    default_detail = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "pred": 0,
        "gold": 0,
        "tp": 0,
    }
    for pid in sorted(id2predicate.keys(), key=lambda x: int(x)):
        p = id2predicate[pid]
        m = detail_metrics.get(p, default_detail)
        # 避免多处引用同一个默认字典
        if m is default_detail:
            m = default_detail.copy()
        per_class_details.append((p, m))
        print(
            f"{p}\t精确率:{m['precision']:.4f}\t召回率:{m['recall']:.4f}\tF1:{m['f1']:.4f}"
            f"\t预测数量:{m['pred']}\t真实数量:{m['gold']}\t正确数量:{m['tp']}"
        )

    print("总体指标：")
    print(
        f"Micro平均\t精确率:{precision:.4f}\t召回率:{recall:.4f}\tF1:{f1:.4f}"
    )
    print(
        f"Macro平均\t精确率:{macro_p:.4f}\t召回率:{macro_r:.4f}\tF1:{macro_f1:.4f}"
    )
    print(
        f"total\t精确率:{precision:.4f}\t召回率:{recall:.4f}\tF1:{f1:.4f}"
    )
    # print(f"一共测试了{total}个数据，成功{success}，失败{fail}")
    data_items = sorted(data_metrics.items(), key=lambda item: item[0])
    data_success_terms = [str(stats["success"]) for _, stats in data_items if stats["total"] > 0]
    data_total_terms = [str(stats["total"]) for _, stats in data_items if stats["total"] > 0]
    data_success_sum = sum(stats["success"] for _, stats in data_items)
    data_total_sum = sum(stats["total"] for _, stats in data_items)
    data_success_expr = "+".join(data_success_terms) if data_success_terms else "0"
    data_total_expr = "+".join(data_total_terms) if data_total_terms else "0"
    data_accuracy = data_success_sum / data_total_sum if data_total_sum else 0.0
    # print(
    #     f"数据判断Accuracy = {{成功{{{data_success_expr}={data_success_sum}}}}}/{{总数{{{data_total_expr}={data_total_sum}}}}} = {data_accuracy:.4f}"
    # )
    valid_details = [
        (name, stats)
        for name, stats in per_class_details
        if stats["gold"] or stats["tp"] or stats["pred"]
    ]
    if not valid_details:
        if per_class_details:
            valid_details = per_class_details
        else:
            valid_details = [("N/A", default_detail)]
    success_terms = [str(stats["tp"]) for _, stats in valid_details]
    total_terms = [str(stats["gold"]) for _, stats in valid_details]
    success_sum = sum(stats["tp"] for _, stats in valid_details)
    total_sum = sum(stats["gold"] for _, stats in valid_details)
    success_expr = "+".join(success_terms) if success_terms else "0"
    total_expr = "+".join(total_terms) if total_terms else "0"
    accuracy = success_sum / total_sum if total_sum else 0.0
    print(
        f"实体判断Accuracy = {{成功{{{success_expr}={success_sum}}}}}/{{总数{{{total_expr}={total_sum}}}}} = {accuracy:.4f}"
    )
    print(
        f"测试集文件输出至{result_dir}目录下{os.path.basename(test_groundtruth_path)}"
    )
    print(
        f"模型预测文件输出至{result_dir}目录下{os.path.basename(test_result_path)}"
    )


def smoke_test(args):
    app_config = load_config(args.config)
    logger = setup_logger(app_config)
    logger.debug("开始烟测")
    set_seed(args.seed)

    label_list = get_label_list(app_config)
    id2label, label2id = build_label_maps(label_list)
    predicate2id = {"喜欢": 0}
    id2predicate = {"0": "喜欢"}

    sample = {
        "text": "张三喜欢李四",
        "triple_list": [("张三", "喜欢", "李四")],
    }
    tokenizer = SimpleTokenizer()

    args.max_len = max(args.max_len, 16)
    args.batch_size = 1
    args.test_batch_size = 1

    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
    )
    config.num_p = len(id2predicate)
    config.num_label = len(label_list)
    config.rounds = 1
    config.fix_bert_embeddings = False
    logger.debug(
        "烟测配置：vocab_size=%d num_p=%d num_label=%d",
        config.vocab_size,
        config.num_p,
        config.num_label,
    )

    model = GRTE(config)
    model.eval()

    dataloader = data_generator(
        args,
        [sample],
        tokenizer,
        [predicate2id, id2predicate],
        [label2id, id2label],
        args.batch_size,
        random=False,
    )
    batch = next(iter(dataloader))
    batch_token_ids, batch_mask = [torch.tensor(d) for d in batch[:2]]

    with torch.no_grad():
        output = model(batch_token_ids, batch_mask)
    logger.debug("烟测完成，输出shape=%s", tuple(output.shape))
