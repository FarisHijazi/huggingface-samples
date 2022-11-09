import numpy as np
import torch
import random

SEPS = ["<,>", "<&>", "<|>", "<;>", "< >"]


def join_subtokens(ner_results):
    joined = []
    for i, result in enumerate(ner_results):
        if result["entity"] == "O":
            joined.append(result)
        elif i > 0 and result["entity"] == ner_results[i - 1]["entity"]:
            joined[-1]["word"] = joined[-1]["word"].lstrip("##") + result[
                "word"
            ].lstrip("##")
            joined[-1]["start"] = min(joined[-1]["start"], ner_results[i - 1]["start"])
            joined[-1]["index"] = min(joined[-1]["index"], ner_results[i - 1]["index"])
            joined[-1]["end"] = max(joined[-1]["end"], ner_results[i - 1]["end"])
        else:
            joined.append(result)
    return joined


def preprocess_string(s):
    for sep in SEPS:
        s = s.replace(sep[1], sep)
    return s


def join_models(strings):
    sep = random.choice(SEPS)
    # if random.random() < 0.3:
    #     sep = sep + ' '
    # if random.random() < 0.3:
    #     sep = ' ' + sep
    # sep = sep.replace('  ', ' ')

    segs = []
    final_string = ""
    pointer = 0
    for i, string in enumerate(strings):
        # replace separators in the string
        string = preprocess_string(string)

        string_delta = string
        if i < len(strings) - 1:
            string_delta += sep
        final_string += string_delta
        segs.append((pointer, pointer + len(string)))
        pointer += len(string_delta)
    return final_string, segs, sep
    # return sep.join(strings)


def encode_tags(labels, encodings):
    # labels = [[tag2id[tag] for tag in doc] for doc in tags]
    # print("labels ", len(labels))
    encoded_labels = []
    for i, (doc_labels, doc_offset) in enumerate(zip(labels, encodings.offset_mapping)):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def get_mixed_example(df, max_items_per_row=2):
    samples = df.sample(1 + int(random.random() * min(max_items_per_row, len(df))))[
        "ModelNo"
    ].values
    joint_string, segs, sep = join_models(samples)
    # joint_string

    token_segs = []
    tokens = []
    pointer = 0
    tags = []
    for i, (start, end) in enumerate(segs):
        new_tokens = [(joint_string[start:end])]
        tags += ["modelno"] * len(new_tokens)
        # FIXME: support multiple tags
        if i < len(segs) - 1:
            sep_tokens = [(sep)]
            tags += ["O"] * len(new_tokens)
            new_tokens += sep_tokens
        tokens += new_tokens
        pointer += len(new_tokens)

    # print(len(tokens), len(tags), tokens, tags)
    assert len(tokens) == len(tags)
    return [tokens, tags]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    new_labels = []
    new_predictions = []
    for (lbl, pred) in zip(labels, predictions):
        new_labels.append([])
        new_predictions.append([])
        for (l, p) in zip(lbl, pred):
            if p != -100 and l != -100:
                new_labels[-1].append(id2tag[l])
                new_predictions[-1].append(id2tag[p])

    # seqeval.f1_score(new_labels, new_predictions)
    seqeval_result = seqeval.compute(predictions=new_predictions, references=new_labels)
    seqeval_result = {f"seqeval_{k}": v for k, v in seqeval_result.items()}
    for k in seqeval_result.get("MISC", {}):
        seqeval_result[f"seqeval.MISC.{k}"] = seqeval_result["MISC"][k]
    if "MISC" in seqeval_result:
        del seqeval_result["MISC"]
    for k in seqeval_result.get("PER", {}):
        seqeval_result[f"seqeval.PER.{k}"] = seqeval_result["PER"][k]
    if "PER" in seqeval_result:
        del seqeval_result["PER"]

    return {
        # **acc.compute(predictions=predictions.reshape(-1), references=labels.reshape(-1)),
        # **f1.compute(predictions=predictions.reshape(-1), references=labels.reshape(-1)),
        **seqeval_result
    }


class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
