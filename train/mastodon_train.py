import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2TokenizerFast, DebertaV2Model, AdamW
from transformers.models.bert.modeling_bert import BertLayer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time
from tqdm import tqdm
import numpy as np
import random
import torch.backends.cudnn as cudnn  # 导入 cudnn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保卷积等操作的一致性
    cudnn.deterministic = True
    cudnn.benchmark = False


# 定义自定义的评估函数
def evalDA(haty, goldy):
    nclasses = 15  # 根据您的数据集调整类别数量
    nok = [0.] * nclasses  # 每个类别的正确预测数
    nrec = [0.] * nclasses  # 每个类别的预测总数
    ntot = [0.] * nclasses  # 每个类别的实际总数
    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        ntot[gldy] += 1
        nrec[recy] += 1
        if recy == gldy:
            nok[gldy] += 1

    nsamps = sum(ntot)
    preval = [float(ntot[i]) / float(nsamps) for i in range(nclasses)]
    prec = 0.
    reca = 0.
    raweval = "DAraweval "
    for j in range(nclasses):
        tp = nok[j]
        pr, re = 0., 0.
        if nrec[j] > 0:
            pr = float(tp) / float(nrec[j])
        if ntot[j] > 0:
            re = float(tp) / float(ntot[j])
        raweval += f"Class {j}: P={pr:.4f}, R={re:.4f} | "
        prec += pr * preval[j]
        reca += re * preval[j]
    print(raweval)
    if prec + reca == 0.:
        f1 = 0.
    else:
        f1 = 2. * prec * reca / (prec + reca)
    return prec, reca, f1


def evalSE(haty, goldy):
    nclasses = 3  # 假设情感类别有3个：中性、积极、消极
    nok = [0.] * nclasses
    nrec = [0.] * nclasses
    ntot = [0.] * nclasses
    for i in range(len(haty)):
        recy = haty[i]
        gldy = goldy[i]
        ntot[gldy] += 1
        nrec[recy] += 1
        if recy == gldy:
            nok[gldy] += 1

    raweval = "SEraweval "
    f1pos, f1neg = 0., 0.
    pr_pos, re_pos = 0., 0.
    pr_neg, re_neg = 0., 0.
    # 计算积极情感的指标（假设标签1代表积极情感）
    for j in (1,):
        tp = nok[j]
        pr, re = 0., 0.
        if nrec[j] > 0:
            pr = float(tp) / float(nrec[j])
        if ntot[j] > 0:
            re = float(tp) / float(ntot[j])
        pr_pos, re_pos = pr, re
        raweval += f"Positive: P={pr:.4f}, R={re:.4f} | "
        if pr + re > 0.:
            f1pos = 2. * pr * re / (pr + re)
    # 计算消极情感的指标（假设标签2代表消极情感）
    for j in (2,):
        tp = nok[j]
        pr, re = 0., 0.
        if nrec[j] > 0:
            pr = float(tp) / float(nrec[j])
        if ntot[j] > 0:
            re = float(tp) / float(ntot[j])
        pr_neg, re_neg = pr, re
        raweval += f"Negative: P={pr:.4f}, R={re:.4f} | "
        if pr + re > 0.:
            f1neg = 2. * pr * re / (pr + re)
    print(raweval)
    # 计算平均精确率和召回率
    avg_prec = (pr_pos + pr_neg) / 2.
    avg_reca = (re_pos + re_neg) / 2.
    avg_f1 = (f1pos + f1neg) / 2.
    return avg_prec, avg_reca, avg_f1


# 读取和处理数据，限制历史对话为最多4句
def read_erc(input_path, tokenizer):
    data_list = []
    with open(input_path, encoding="utf-8") as file_read:
        data = json.load(file_read)

        for dialogues in data:
            if len(dialogues) == 0:
                continue

            history = []
            speakers_so_far = []
            for i, utterance_data in enumerate(dialogues):
                current_speaker = utterance_data['speaker']
                utterance = utterance_data['utterance']
                sentiment = utterance_data['sentiment']
                act = utterance_data['act']

                # 更新历史对话（包括说话人），只保留最近的4句
                if len(history) >= 6:
                    history.pop(0)
                    speakers_so_far.pop(0)
                history.append(f"{current_speaker}: {utterance}")
                speakers_so_far.append(current_speaker)

                # 构建当前对话的文本，包含最多前面四句历史对话
                input_text = f" {tokenizer.sep_token} ".join(history[-6:])

                # 当前话语在历史中的索引（由于只保留4句，需要调整索引）
                current_utterance_index = min(i, 5)  # 索引最大为4

                data_list.append({
                    "input_text": input_text,
                    "sentiment": sentiment,
                    "act": act,
                    "speakers": speakers_so_far.copy(),
                    "current_utterance_index": current_utterance_index  # 当前话语在历史中的索引
                })

    return data_list


# 定义数据集
class ConversationDataset(Dataset):
    def __init__(self, texts, sentiments, acts, speakers,
                 current_utterance_indices, importance_labels_list, tokenizer, max_length):
        self.texts = texts
        self.sentiments = sentiments
        self.acts = acts
        self.speakers = speakers  # 列表，每个元素是一个说话人列表
        self.current_utterance_indices = current_utterance_indices
        self.importance_labels_list = importance_labels_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 存储每个样本的 utterance_positions
        self.utterance_positions_list = []

        for text in self.texts:
            # 编码文本，获取每个话语的起始位置
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_offsets_mapping=True,
                padding='max_length',
                truncation=True
            )

            input_ids = encoding['input_ids']

            # 找到每个句子的起始位置（通过 [SEP] 标记）
            sep_token_id = self.tokenizer.sep_token_id
            utterance_positions = [i for i, token_id in enumerate(input_ids)
                                   if token_id == sep_token_id]

            # 在列表开头添加 [CLS] 位置 (0)
            utterance_positions = [0] + [pos + 1 for pos in utterance_positions]

            # 存储起始位置
            self.utterance_positions_list.append(utterance_positions)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        act = self.acts[idx]
        speakers = self.speakers[idx]  # 列表
        current_utterance_index = self.current_utterance_indices[idx]
        importance_labels = self.importance_labels_list[idx]
        importance_scores = importance_labels['importance_scores']  # 原始长度
        importance_mask = importance_labels['importance_mask']  # 原始长度

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # 获取当前样本的 utterance_positions
        utterance_positions = torch.tensor(
            self.utterance_positions_list[idx], dtype=torch.long
        )

        # 调整说话人列表，使其与实际的句子数量对齐
        num_utterances = min(len(utterance_positions), len(speakers))
        utterance_positions = utterance_positions[:num_utterances]
        speakers = speakers[:num_utterances]

        # 将 speakers 转换为数值索引的 tensor
        speakers = torch.tensor(speakers, dtype=torch.long)

        # 当前话语在 utterance_positions 中的索引
        # 确保 current_utterance_index 不超过 num_utterances-1
        if num_utterances > 0:
            current_utterance_position_index = min(
                current_utterance_index, num_utterances - 1
            )
        else:
            current_utterance_position_index = 0

        # 转换 importance_scores 和 importance_mask 为 tensors
        importance_scores = torch.tensor(importance_scores, dtype=torch.float)
        importance_mask = torch.tensor(importance_mask, dtype=torch.float)

        # 调整 importance_scores 和 importance_mask 的长度
        importance_length = importance_scores.size(0)
        if importance_length > num_utterances:
            # 截断
            importance_scores = importance_scores[:num_utterances]
            importance_mask = importance_mask[:num_utterances]
        elif importance_length < num_utterances:
            # 填充
            padding_size = num_utterances - importance_length
            scores_padding = torch.zeros(padding_size, dtype=torch.float)
            mask_padding = torch.zeros(padding_size, dtype=torch.float)
            importance_scores = torch.cat([importance_scores, scores_padding], dim=0)
            importance_mask = torch.cat([importance_mask, mask_padding], dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'act': torch.tensor(act, dtype=torch.long),
            'speakers': speakers,  # 转换为 tensor
            'utterance_positions': utterance_positions,
            'current_utterance_position_index':
                torch.tensor(current_utterance_position_index, dtype=torch.long),
            'importance_scores': importance_scores,
            'importance_mask': importance_mask
        }


# 定义多任务模型，并融合说话人模块和历史重要性模块
class MultiTaskDeberta(torch.nn.Module):
    def __init__(self, model_name, num_sentiment_labels, num_act_labels):
        super(MultiTaskDeberta, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained(
            model_name, output_hidden_states=True
        )

        self.num_sentiment_labels = num_sentiment_labels
        self.num_act_labels = num_act_labels

        # 定义分类器
        self.sentiment_classifier = torch.nn.Linear(
            self.deberta.config.hidden_size, num_sentiment_labels
        )
        self.act_classifier = torch.nn.Linear(
            self.deberta.config.hidden_size, num_act_labels
        )

        # 定义检测说话人关系和历史重要性关系的线性层
        self.speaker_relation_detector = torch.nn.Linear(
            self.deberta.config.hidden_size * 4, 1
        )
        self.importance_relation_detector = torch.nn.Linear(
            self.deberta.config.hidden_size * 4, 1
        )

        # 定义 MHA (Multi-Head Attention) 层用于说话人和重要性模块
        self.speaker_mha_layers = 2
        self.importance_mha_layers = 2
        self.config = self.deberta.config

        # 定义说话人多头注意力层
        for i in range(self.speaker_mha_layers):
            layer = BertLayer(self.config)
            self.add_module(f"Speaker_MHA_{i}", layer)

        # 定义重要性多头注意力层
        for i in range(self.importance_mha_layers):
            layer = BertLayer(self.config)
            self.add_module(f"Importance_MHA_{i}", layer)

        # 定义融合函数：将基础模型输出与说话人/重要性模块输出融合
        self.fusion_fct_speaker = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            torch.nn.Tanh()
        )
        self.fusion_fct_importance = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            torch.nn.Tanh()
        )

    def forward(self, input_ids, attention_mask, sentiment_labels=None,
                act_labels=None, speakers=None, utterance_positions=None,
                current_utterance_position_index=None,
                importance_scores=None, importance_mask=None):
        device = input_ids.device
        # 编码输入，获取隐藏状态
        outputs = self.deberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        all_hidden_states = outputs.hidden_states  # Tuple of layers

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        hidden_size = hidden_states.size(-1)

        # 提取 speaker_hidden_states 和 importance_hidden_states
        speaker_hidden_states = all_hidden_states[-(self.speaker_mha_layers + 1)].detach()
        importance_hidden_states = all_hidden_states[-(self.importance_mha_layers + 1)]

        # 初始化列表以收集结果
        fused_hidden_states_list = []
        speaker_logits_list = []
        same_speaker_labels_list = []
        importance_loss_list = []
        importance_logits_list = []
        importance_labels_list = []

        # 迭代处理批次中的每个样本
        for i in range(batch_size):
            # 获取当前样本的数据
            input_ids_i = input_ids[i]
            attention_mask_i = attention_mask[i]
            hidden_states_i = hidden_states[i]  # (seq_len, hidden_size)
            speaker_hidden_states_i = speaker_hidden_states[i]  # (seq_len, hidden_size)
            importance_hidden_states_i = importance_hidden_states[i]  # (seq_len, hidden_size)
            speakers_i = speakers[i].to(device)
            utterance_positions_i = utterance_positions[i].to(device)
            current_idx = current_utterance_position_index[i].item()

            # 扩展 attention mask
            attention_mask_i_expanded = attention_mask_i.unsqueeze(0).unsqueeze(0)
            attention_mask_i_expanded = attention_mask_i_expanded.to(dtype=next(self.parameters()).dtype)
            attention_mask_i_expanded = (1.0 - attention_mask_i_expanded) * -10000.0  # (1, 1, seq_len)

            # 说话人多头注意力模块
            speaker_hidden_states_i = speaker_hidden_states_i.unsqueeze(0)  # (1, seq_len, hidden_size)
            for j in range(self.speaker_mha_layers):
                mha_layer = getattr(self, f"Speaker_MHA_{j}")
                layer_outputs = mha_layer(hidden_states=speaker_hidden_states_i,
                                          attention_mask=attention_mask_i_expanded)
                speaker_hidden_states_i = layer_outputs[0]

            speaker_hidden_states_i = speaker_hidden_states_i.squeeze(0)  # (seq_len, hidden_size)

            # 说话人关系检测
            num_utterances = utterance_positions_i.size(0)
            if num_utterances > 1:
                positions = utterance_positions_i  # (num_utterances,)
                utterance_embs = speaker_hidden_states_i[positions, :]  # (num_utterances, hidden_size)
                current_utterance_emb = utterance_embs[current_idx]  # (hidden_size,)

                # 排除当前话语嵌入
                other_indices = list(range(num_utterances))
                other_indices.remove(current_idx)
                other_utterance_embs = utterance_embs[other_indices, :]  # (num_other, hidden_size)

                # 计算说话人关系特征
                current_emb_expanded = current_utterance_emb.unsqueeze(0).expand(other_utterance_embs.size(0), -1)
                speaker_features = torch.cat([
                    other_utterance_embs,
                    current_emb_expanded,
                    other_utterance_embs * current_emb_expanded,
                    other_utterance_embs - current_emb_expanded
                ], dim=-1)  # (num_other, hidden_size * 4)

                speaker_logit = self.speaker_relation_detector(speaker_features).squeeze(-1)  # (num_other,)
                speaker_logits_list.append(speaker_logit)

                # 计算相同说话人标签
                masked_speaker = speakers_i[current_idx]
                other_speakers = speakers_i[other_indices]
                same_speaker_labels = (other_speakers == masked_speaker).float()
                same_speaker_labels_list.append(same_speaker_labels)
            else:
                # 如果只有一个话语，跳过
                speaker_logit = torch.tensor([], device=device)
                same_speaker_labels = torch.tensor([], device=device)
                speaker_logits_list.append(speaker_logit)
                same_speaker_labels_list.append(same_speaker_labels)

            # 融合隐藏状态和说话人模块输出
            fused_hidden_states_speaker_i = self.fusion_fct_speaker(
                torch.cat([
                    hidden_states_i,
                    speaker_hidden_states_i,
                    hidden_states_i * speaker_hidden_states_i,
                    hidden_states_i - speaker_hidden_states_i
                ], dim=-1)
            )  # (seq_len, hidden_size)

            # 重要性多头注意力模块
            importance_hidden_states_i = importance_hidden_states_i.unsqueeze(0)  # (1, seq_len, hidden_size)
            for j in range(self.importance_mha_layers):
                mha_layer = getattr(self, f"Importance_MHA_{j}")
                layer_outputs = mha_layer(hidden_states=importance_hidden_states_i,
                                          attention_mask=attention_mask_i_expanded)
                importance_hidden_states_i = layer_outputs[0]

            importance_hidden_states_i = importance_hidden_states_i.squeeze(0)  # (seq_len, hidden_size)

            # 重要性关系检测
            if num_utterances > 1:
                positions = utterance_positions_i  # (num_utterances,)
                utterance_embs = importance_hidden_states_i[positions, :]  # (num_utterances, hidden_size)
                current_utterance_emb = utterance_embs[current_idx]  # (hidden_size,)

                # 仅考虑历史话语
                history_indices = list(range(current_idx))  # 历史话语索引
                if len(history_indices) > 0:
                    history_utterance_embs = utterance_embs[history_indices, :]  # (num_history, hidden_size)

                    current_emb_expanded = current_utterance_emb.unsqueeze(0).expand(history_utterance_embs.size(0), -1)
                    importance_features = torch.cat([
                        history_utterance_embs,
                        current_emb_expanded,
                        history_utterance_embs * current_emb_expanded,
                        history_utterance_embs - current_emb_expanded
                    ], dim=-1)  # (num_history, hidden_size * 4)

                    importance_logits = self.importance_relation_detector(importance_features).squeeze(
                        -1)  # (num_history,)
                    importance_logits_list.append(importance_logits)

                    # 计算重要性损失
                    importance_scores_i = importance_scores[i].to(device)
                    importance_mask_i = importance_mask[i].to(device)
                    importance_scores_i = importance_scores_i[history_indices]
                    importance_mask_i = importance_mask_i[history_indices]

                    importance_loss_fct = torch.nn.MSELoss(reduction='none')
                    importance_loss_i = importance_loss_fct(importance_logits, importance_scores_i)
                    masked_importance_loss = (
                                                         importance_loss_i * importance_mask_i).sum() / importance_mask_i.sum().clamp(
                        min=1e-8)
                    importance_loss_list.append(masked_importance_loss)
                    importance_labels_list.append(importance_scores_i)
                else:
                    importance_loss_list.append(torch.tensor(0.0, device=device))
                    importance_logits_list.append(torch.tensor([], device=device))
                    importance_labels_list.append(torch.tensor([], device=device))
            else:
                importance_loss_list.append(torch.tensor(0.0, device=device))
                importance_logits_list.append(torch.tensor([], device=device))
                importance_labels_list.append(torch.tensor([], device=device))

            # 融合隐藏状态和重要性模块输出
            fused_hidden_states_i = self.fusion_fct_importance(
                torch.cat([
                    fused_hidden_states_speaker_i,
                    importance_hidden_states_i,
                    fused_hidden_states_speaker_i * importance_hidden_states_i,
                    fused_hidden_states_speaker_i - importance_hidden_states_i
                ], dim=-1)
            )  # (seq_len, hidden_size)

            # 添加融合后的隐藏状态到列表
            fused_hidden_states_list.append(fused_hidden_states_i)

        # 将列表转换为张量
        fused_hidden_states = torch.stack(fused_hidden_states_list, dim=0)  # (batch_size, seq_len, hidden_size)

        # 获取当前话语的位置，用于分类
        current_utterance_positions = []
        for i in range(batch_size):
            if utterance_positions[i].numel() == 0:
                current_utterance_positions.append(torch.tensor(0, device=input_ids.device))
            else:
                idx = current_utterance_position_index[i].item()
                if idx >= utterance_positions[i].size(0):
                    idx = utterance_positions[i].size(0) - 1
                current_pos = utterance_positions[i][idx]
                current_utterance_positions.append(current_pos)
        current_utterance_positions = torch.stack(current_utterance_positions).to(input_ids.device)

        # 防止索引越界
        current_utterance_positions = torch.clamp(current_utterance_positions, max=fused_hidden_states.size(1) - 1)

        pooled_output = fused_hidden_states[
                        torch.arange(batch_size, device=input_ids.device), current_utterance_positions, :
                        ]

        sentiment_logits = self.sentiment_classifier(pooled_output)
        act_logits = self.act_classifier(pooled_output)

        # 计算损失
        total_loss = None
        if sentiment_labels is not None and act_labels is not None:
            # 情感和行为预测使用 CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            act_loss = loss_fct(act_logits, act_labels)

            # 说话人预测损失
            if len(speaker_logits_list) > 0:
                speaker_logits_cat = torch.cat(speaker_logits_list, dim=0)
                if speaker_logits_cat.numel() > 0:
                    speaker_labels_cat = torch.cat(same_speaker_labels_list, dim=0).to(device)
                    bce_loss_fct = torch.nn.BCEWithLogitsLoss()
                    speaker_loss = bce_loss_fct(speaker_logits_cat, speaker_labels_cat)
                else:
                    speaker_loss = torch.tensor(0.0, device=device)
            else:
                speaker_loss = torch.tensor(0.0, device=device)

            # 重要性预测损失
            if len(importance_loss_list) > 0:
                importance_loss = torch.stack(importance_loss_list).mean()
            else:
                importance_loss = torch.tensor(0.0, device=device)

            # 聚合所有损失
            total_loss = sentiment_loss + act_loss + 0.6 * speaker_loss + 0.4 * importance_loss

        return total_loss, sentiment_logits, act_logits, speaker_logits_list, same_speaker_labels_list, importance_logits_list, importance_labels_list


# 修改评估函数，使用自定义的 evalDA 和 evalSE
def evaluate(model, data_loader, device):
    model.eval()
    all_sentiment_preds = []
    all_act_preds = []
    all_sentiment_labels = []
    all_act_labels = []
    all_speaker_preds = []
    all_speaker_labels = []
    all_importance_preds = []
    all_importance_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            act_labels = batch['act'].to(device)
            speakers = batch['speakers']
            utterance_positions = batch['utterance_positions']
            current_utterance_position_index = batch['current_utterance_position_index'].to(device)

            importance_scores = batch['importance_scores']
            importance_mask = batch['importance_mask']

            if importance_scores is not None:
                importance_scores = [score.to(device) for score in importance_scores]
            if importance_mask is not None:
                importance_mask = [mask.to(device) for mask in importance_mask]

            outputs = model(
                input_ids,
                attention_mask,
                sentiment_labels=None,
                act_labels=None,
                speakers=speakers,
                utterance_positions=utterance_positions,
                current_utterance_position_index=current_utterance_position_index,
                importance_scores=importance_scores,
                importance_mask=importance_mask
            )

            # 解包 model.forward 返回的结果
            _, sentiment_logits, act_logits, speaker_logits_list, same_speaker_labels_list, importance_logits_list, importance_labels_list = outputs

            # 计算情感和行为预测
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            act_preds = torch.argmax(act_logits, dim=1).cpu().numpy()

            all_sentiment_preds.extend(sentiment_preds)
            all_act_preds.extend(act_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_act_labels.extend(act_labels.cpu().numpy())

            # 处理说话人预测
            if len(speaker_logits_list) > 0:
                speaker_logits_cat = torch.cat(speaker_logits_list, dim=0)
                if speaker_logits_cat.numel() > 0:
                    speaker_labels_cat = torch.cat(same_speaker_labels_list, dim=0).to(device)
                    speaker_preds = (torch.sigmoid(speaker_logits_cat) > 0.5).int().cpu().numpy()
                    speaker_labels = speaker_labels_cat.cpu().numpy()

                    all_speaker_preds.extend(speaker_preds)
                    all_speaker_labels.extend(speaker_labels)

            # 处理历史重要性预测
            if importance_logits_list:
                for importance_preds, importance_scores_tensor in zip(importance_logits_list, importance_labels_list):
                    if importance_preds.numel() > 0:
                        importance_preds = importance_preds.cpu().numpy()
                        importance_scores = importance_scores_tensor.cpu().numpy()
                        all_importance_preds.extend(importance_preds)
                        all_importance_labels.extend(importance_scores)

    # 将标签和预测转换为列表
    all_act_labels = list(all_act_labels)
    all_act_preds = list(all_act_preds)
    all_sentiment_labels = list(all_sentiment_labels)
    all_sentiment_preds = list(all_sentiment_preds)

    # 将情感标签映射到 0（中性）、1（积极）、2（消极）
    sentiment_mapping = {'*': 0, '+': 1, '-': 2}
    inverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}

    all_sentiment_labels_mapped = [sentiment_mapping[sentiment_encoder.inverse_transform([label])[0]] for label in
                                   all_sentiment_labels]
    all_sentiment_preds_mapped = [sentiment_mapping[sentiment_encoder.inverse_transform([pred])[0]] for pred in
                                  all_sentiment_preds]

    # 使用自定义的评估函数
    prec_act, reca_act, f1_act = evalDA(all_act_preds, all_act_labels)
    prec_sentiment, reca_sentiment, f1_sentiment = evalSE(all_sentiment_preds_mapped, all_sentiment_labels_mapped)

    # 计算说话人预测的准确率
    if all_speaker_labels:
        speaker_correct = np.sum(np.array(all_speaker_preds) == np.array(all_speaker_labels))
        speaker_total = len(all_speaker_labels)
        speaker_accuracy = speaker_correct / speaker_total
    else:
        speaker_accuracy = 0.0

    # 计算历史重要性预测的平均 MSE
    if all_importance_labels:
        mse_loss = np.mean((np.array(all_importance_preds) - np.array(all_importance_labels)) ** 2)
    else:
        mse_loss = 0.0

    # 返回所需的指标
    return {
        'act': {
            'precision': prec_act,
            'recall': reca_act,
            'f1': f1_act
        },
        'sentiment': {
            'precision': prec_sentiment,
            'recall': reca_sentiment,
            'f1': f1_sentiment
        },
        'speaker': {
            'accuracy': speaker_accuracy
        },
        'importance': {
            'mse_loss': mse_loss
        }
    }


if __name__ == "__main__":
    set_seed(42)
    # 数据路径
    train_path = '/public/home/acy7tl31mi/demo2-classification/dataset/mastodon/train-copy.json'
    dev_path = '/public/home/acy7tl31mi/demo2-classification/dataset/mastodon/test-copy.json'

    # 加载重要性标签
    with open(
            '/public/home/acy7tl31mi/demo2-classification/classify-deberta/mastodon-train_importance_labels-debertav2xl-sixhistory.json',
            'r', encoding='utf-8') as f:
        train_importance_labels = json.load(f)
    with open(
            '/public/home/acy7tl31mi/demo2-classification/classify-deberta/mastodon-test_importance_labels-debertav2xl-sixhistory.json',
            'r', encoding='utf-8') as f:
        dev_importance_labels = json.load(f)

    # 使用 DebertaV2TokenizerFast
    tokenizer = DebertaV2TokenizerFast.from_pretrained(
        "/public/home/acy7tl31mi/demo2-classification/classify-deberta/trainmodel-deberta-v3-base"
    )

    # 读取数据
    train_data = read_erc(train_path, tokenizer)
    dev_data = read_erc(dev_path, tokenizer)

    # 将文本和标签分开
    train_texts = [item['input_text'] for item in train_data]
    train_sentiments = [item['sentiment'] for item in train_data]
    train_acts = [item['act'] for item in train_data]
    train_speakers = [item['speakers'] for item in train_data]
    train_current_utterance_indices = [item['current_utterance_index'] for item in train_data]

    dev_texts = [item['input_text'] for item in dev_data]
    dev_sentiments = [item['sentiment'] for item in dev_data]
    dev_acts = [item['act'] for item in dev_data]
    dev_speakers = [item['speakers'] for item in dev_data]
    dev_current_utterance_indices = [item['current_utterance_index'] for item in dev_data]

    # 标签编码
    sentiment_encoder = LabelEncoder()
    act_encoder = LabelEncoder()
    speaker_encoder = LabelEncoder()

    # 对情感和行为标签进行编码
    train_sentiments_encoded = sentiment_encoder.fit_transform(train_sentiments)
    train_acts_encoded = act_encoder.fit_transform(train_acts)

    dev_sentiments_encoded = sentiment_encoder.transform(dev_sentiments)
    dev_acts_encoded = act_encoder.transform(dev_acts)

    # 收集所有的说话人标签并进行编码
    all_speakers = set()
    for spk in train_speakers + dev_speakers:
        all_speakers.update(spk)
    speaker_encoder.fit(list(all_speakers))

    # 对说话人标签进行编码
    train_speakers_encoded = []
    for spk in train_speakers:
        # 将每个说话人转换为相应的编码
        spk_indices = speaker_encoder.transform(spk)
        train_speakers_encoded.append(spk_indices)

    dev_speakers_encoded = []
    for spk in dev_speakers:
        spk_indices = speaker_encoder.transform(spk)
        dev_speakers_encoded.append(spk_indices)

    np.save('sentiment_classes.npy', sentiment_encoder.classes_)
    np.save('act_classes.npy', act_encoder.classes_)
    np.save('speaker_classes.npy', speaker_encoder.classes_)

    # 创建数据集和数据加载器
    BATCH_SIZE = 8
    MAX_LENGTH = 384

    train_dataset = ConversationDataset(
        texts=train_texts,
        sentiments=train_sentiments_encoded,
        acts=train_acts_encoded,
        speakers=train_speakers_encoded,
        current_utterance_indices=train_current_utterance_indices,
        importance_labels_list=train_importance_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    dev_dataset = ConversationDataset(
        texts=dev_texts,
        sentiments=dev_sentiments_encoded,
        acts=dev_acts_encoded,
        speakers=dev_speakers_encoded,
        current_utterance_indices=dev_current_utterance_indices,
        importance_labels_list=dev_importance_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )


    def collate_fn(batch):
        # 自定义 collate_fn，用于处理不同长度的 utterance_positions 和 speakers
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        sentiment = torch.stack([item['sentiment'] for item in batch])
        act = torch.stack([item['act'] for item in batch])
        current_utterance_position_index = torch.stack(
            [item['current_utterance_position_index'] for item in batch]
        )

        # 将 utterance_positions 和 speakers 放在列表中，以保留各自的长度
        utterance_positions = [item['utterance_positions'] for item in batch]
        speakers = [item['speakers'] for item in batch]
        importance_scores = [item['importance_scores'] for item in batch]
        importance_mask = [item['importance_mask'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentiment': sentiment,
            'act': act,
            'speakers': speakers,
            'utterance_positions': utterance_positions,
            'current_utterance_position_index': current_utterance_position_index,
            'importance_scores': importance_scores,
            'importance_mask': importance_mask
        }


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    # 实例化模型
    model_name = "/public/home/acy7tl31mi/demo2-classification/classify-deberta/trainmodel-deberta-v3-base"
    num_sentiment_labels = len(sentiment_encoder.classes_)
    num_act_labels = len(act_encoder.classes_)
    model = MultiTaskDeberta(model_name, num_sentiment_labels, num_act_labels)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 训练模型
    EPOCHS = 150
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    best_avg_f1 = 0.0
    best_act_f1 = 0.0
    best_sentiment_f1 = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_batches = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            act_labels = batch['act'].to(device)
            speakers = batch['speakers']
            utterance_positions = batch['utterance_positions']
            current_utterance_position_index = batch['current_utterance_position_index'].to(device)
            importance_scores = batch['importance_scores']
            importance_mask = batch['importance_mask']

            if importance_scores is not None:
                importance_scores = [score.to(device) for score in importance_scores]
            if importance_mask is not None:
                importance_mask = [mask.to(device) for mask in importance_mask]

            outputs = model(
                input_ids,
                attention_mask,
                sentiment_labels,
                act_labels,
                speakers=speakers,
                utterance_positions=utterance_positions,
                current_utterance_position_index=current_utterance_position_index,
                importance_scores=importance_scores,
                importance_mask=importance_mask
            )

            loss = outputs[0]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, '
              f'Time: {epoch_time:.2f}s')

        metrics = evaluate(model, dev_loader, device)

        print(f'Epoch {epoch + 1}/{EPOCHS} Evaluation:')
        print(f'Dialogue Act Recognition (DAR) - Precision: '
              f'{metrics["act"]["precision"]:.4f}, Recall: '
              f'{metrics["act"]["recall"]:.4f}, F1-score: '
              f'{metrics["act"]["f1"]:.4f}')
        print(f'Sentiment Recognition (SR) - Precision: '
              f'{metrics["sentiment"]["precision"]:.4f}, Recall: '
              f'{metrics["sentiment"]["recall"]:.4f}, F1-score: '
              f'{metrics["sentiment"]["f1"]:.4f}')
        print(f'Speaker Prediction - Accuracy: '
              f'{metrics["speaker"]["accuracy"]:.4f}')
        print(f'Importance Prediction - MSE Loss: '
              f'{metrics["importance"]["mse_loss"]:.4f}')

        sentiment_f1 = metrics["sentiment"]["f1"]
        act_f1 = metrics["act"]["f1"]
        avg_f1 = (sentiment_f1 + act_f1) / 2

        # 检查三个条件，满足任意一个就保存模型
        save_model = False
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            save_model = True
            print(f"Average F1 improved to {best_avg_f1:.4f}")
        if sentiment_f1 > best_sentiment_f1:
            best_sentiment_f1 = sentiment_f1
            save_model = True
            print(f"Sentiment F1 improved to {best_sentiment_f1:.4f}")
        if act_f1 > best_act_f1:
            best_act_f1 = act_f1
            save_model = True
            print(f"Act F1 improved to {best_act_f1:.4f}")

        if save_model:
            torch.save(model.state_dict(),
                       '/public/home/acy7tl31mi/demo2-classification/classify-deberta/bestmodel/mastodon-best_model_deberta-v3-base-train_importance_labels-addspeakerpredict-sixhistory-checkpoint.pth')
            print(f"Best model saved with act F1: {best_act_f1:.4f}, "
                  f"sentiment F1: {best_sentiment_f1:.4f}")

    print(f"Training completed. Best act F1: {best_act_f1:.4f}, "
          f"Best sentiment F1: {best_sentiment_f1:.4f}"

          )
