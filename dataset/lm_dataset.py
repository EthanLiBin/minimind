from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

# PretrainDataset:  整段文本都算 loss，每个位置都要预测下一个 token
class PretrainDataset(Dataset):
    # 数据长什么样？{"text": "中国有五千年灿烂文明，从夏商周到秦汉，从唐宋到明清……"}
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        # 初始化
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 从json文件中加载预训练数据
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 转tokenizer处理
        # 为什么要 max_length - 2？ 因为后面要手动加头尾，先留出两个位置
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 添加 头(bos_token_id) / 尾(eos_token_id) 标记
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 如果一段文本不够长（短于 max_length），右边补 padding
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 制作labels，和input_ids保持一致。在训练时进行向右错1位处理，跟生成的下一个token做比较来训练
        labels = input_ids.clone()
        # label中的padding位置抹掉(-100处理)，padding不算loss
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

# SFTDataset:        只对 assistant 的回复算 loss，用户的话、系统提示都不算
# 在整段对话中画一个圈，只对 assistant 的发言算 loss，其他全是 -100。
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        # 初始化
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 定义数据结构
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        # 加载训练数据
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # assistant 开始的标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # 结束标记
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    # 把原始对话转成模型能看懂的格式：
    # 原始:  [{"role":"user","content":"1+1等于几"}, {"role":"assistant","content":"1+1等于2"}]
    # apply_chat_template 后:
    # "<|im_start|>user\n1+1等于几<|im_end|>\n<|im_start|>assistant\n1+1等于2<|im_end|>\n"
    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    # input_ids:   <|start|> user 1+1等于几 <|end|> <|start|> assistant 1+1等于2 </s>
    # labels:     [-100 -100 -100 -100 -100 -100 -100 -100 -100     1  +  1 等 于 2  </s>]
    #             ↑                                            ↑───────────────────────↑
    #             全忽略                                        只有这部分算 loss
    def generate_labels(self, input_ids):
        # # 全部先置 -100（忽略）
        # -100 是 PyTorch cross_entropy 的 ignore_index，这些位置的 loss 不计入。
        # 然后扫描整个序列，找到打标记 "assistant\n" → "</s>" 的区域：
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到 "assistant\n"，bos_id 就是 assistant\n
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 从 assistant\n 后面开始
                start = i + len(self.bos_id)
                end = start
                # 找到 "</s>"，eos_id就是"</s>"
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 把 [start : end] 这片区域的 labels 设为 actual token ids
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 添加系统信息 system
        conversations = pre_processing_chat(sample['conversations'])
        # 把原始对话转成模型能看懂的格式：
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        # 截断到最大长度
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 不足的补0
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # * 只标记 assistant 的部分
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

# 数据案例：
# {"conversations": [{"role": "user", "content": "你有没有遇到过这样的情况：明明想做一件事，却因为时间或精力不够而放弃？你希望找到一种方法，既能高效完成目标，又能平衡其他生活事务。你愿意和我分享你的困惑吗？"}, {"role": "assistant", "content": "是的，我理解这种困扰。时间与精力的限制常常让我们难以坚持目标。一种有效的方法是制定清晰的计划，优先处理重要任务，并使用时间管理工具如番茄工作法或待办清单来提高效率。同时，学会说“不”，合理分配时间，确保重要事务得到充分关注。平衡生活需要自我觉察与灵活调整，关键在于设定合理目标并保持持续努力。你愿意分享你的具体困扰吗？"}, {"role": "user", "content": "你今天有没有遇到过需要快速决策但又犹豫不决的情况？"}, {"role": "assistant", "content": ""}]}
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 添加system信息
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        # 把原始对话转成模型能看懂的格式：
        # 原始:  [{"role":"user","content":"1+1等于几"}, {"role":"assistant","content":"1+1等于2"}]
        # apply_chat_template 后:
        # "<|im_start|>user\n1+1等于几<|im_end|>\n<|im_start|>assistant\n1+1等于2<|im_end|>\n"
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        # 生成模型能理解的prompt
        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass