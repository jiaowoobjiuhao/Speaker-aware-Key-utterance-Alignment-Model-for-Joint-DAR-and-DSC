import json
import torch
from transformers import DebertaV2Tokenizer, DebertaV2Model
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 读取和处理数据，限制历史对话为最多5句
def read_erc_with_history(input_path):
    data_list = []
    with open(input_path, encoding="utf-8") as file_read:
        data = json.load(file_read)
        for dialogues in data:
            if len(dialogues) == 0:
                continue
            history = []
            for i in range(len(dialogues)):
                current_speaker = dialogues[i]['speaker']
                current_utterance = dialogues[i]['utterance']
                sentiment = dialogues[i]['sentiment']
                act = dialogues[i]['act']

                # 在训练时，需要在话语前添加说话人
                history_with_speaker = history.copy()
                history_with_speaker.append(f"{current_speaker}: {current_utterance}")

                # 准备输入文本，使用 [SEP] 分隔句子
                input_text = " [SEP] ".join(history_with_speaker[-9:])  # 只保留最多前5句历史对话

                # 对于生成重要性评分，我们在这里不包含说话人
                history_without_speaker = [utterance for utterance in history]

                # 只保留最多前5句历史话语
                if len(history_without_speaker) > 9:
                    history_without_speaker = history_without_speaker[-9:]

                current_utterance_no_speaker = current_utterance

                data_list.append({
                    "input_text": input_text,
                    "sentiment": sentiment,
                    "act": act,
                    "history": history_without_speaker,  # 历史话语（不含说话人），最多5句
                    "current_utterance": current_utterance_no_speaker  # 当前话语（不含说话人）
                })

                # 更新历史对话（不含说话人）
                history.append(current_utterance)
                # 确保历史对话不超过5句
                if len(history) > 9:
                    history.pop(0)
    return data_list

# 生成重要性标签并保存到本地
def generate_and_save_importance_labels(data_list, model_name, save_path):
    # 初始化 tokenizer 和模型
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = DebertaV2Model.from_pretrained(model_name)
    model.eval()
    model.to(device)

    importance_labels_list = []

    for idx, item in enumerate(tqdm(data_list, desc="Generating importance labels")):
        history = item['history']  # 历史句子列表（不含说话人），最多5句
        current_utterance = item['current_utterance']  # 当前句子（不含说话人）

        # 对每个历史句子，计算与当前句子的语义相似度
        importance_scores = []
        for prev_utterance in history:
            # 编码历史句子和当前句子
            prev_inputs = tokenizer(prev_utterance, return_tensors='pt', truncation=True, max_length=128).to(device)
            curr_inputs = tokenizer(current_utterance, return_tensors='pt', truncation=True, max_length=128).to(device)

            with torch.no_grad():
                # 获取 [CLS] token 的嵌入
                prev_embedding = model(**prev_inputs).last_hidden_state[:, 0, :]  # (1, hidden_size)
                curr_embedding = model(**curr_inputs).last_hidden_state[:, 0, :]  # (1, hidden_size)

                # 计算余弦相似度
                similarity = torch.nn.functional.cosine_similarity(prev_embedding, curr_embedding).item()
                importance_scores.append(similarity)

        # 归一化重要性得分到 [0, 1]
        importance_scores = np.array(importance_scores)
        if len(importance_scores) > 0:
            min_score = importance_scores.min()
            max_score = importance_scores.max()
            if max_score - min_score > 1e-6:
                normalized_scores = (importance_scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(importance_scores)
        else:
            normalized_scores = np.zeros(0)

        # 创建掩码，指示有效的重要性得分位置
        importance_mask = np.ones_like(normalized_scores, dtype=np.float32)

        importance_labels_list.append({
            'importance_scores': normalized_scores.tolist(),  # 长度为历史句子数，最多4句
            'importance_mask': importance_mask.tolist()       # 长度为历史句子数，最多4句
        })

    # 将重要性标签保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(importance_labels_list, f, ensure_ascii=False)

    print(f"Importance labels saved to {save_path}")

if __name__ == "__main__":
    # 数据路径
    train_path = '/public/home/acy7tl31mi/demo2-classification/dataset/mastodon/train-copy.json'
    dev_path = '/public/home/acy7tl31mi/demo2-classification/dataset/mastodon/dev-copy.json'
    test_path = '/public/home/acy7tl31mi/demo2-classification/dataset/mastodon/test-copy.json'
    # 模型名称（使用 DeBERTa-v2-xlarge）
    model_name = "/public/home/acy7tl31mi/demo2-classification/classify-deberta/trainmodel-deberta-v2-xlarge"

    # 读取数据
    train_data = read_erc_with_history(train_path)
    dev_data = read_erc_with_history(dev_path)
    test_data = read_erc_with_history(test_path)
    # 生成并保存重要性标签
    generate_and_save_importance_labels(train_data, model_name, 'mastodon-train_importance_labels-debertav2xl-ninehistory.json')
    generate_and_save_importance_labels(dev_data, model_name, 'mastodon-dev_importance_labels-debertav2xl-ninehistory.json')
    generate_and_save_importance_labels(test_data, model_name, 'mastodon-test_importance_labels-debertav2xl-ninehistory.json')
