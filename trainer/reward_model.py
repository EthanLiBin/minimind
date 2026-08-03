'''
Author: Ethan
Date: 2026-08-02 15:19:56
LastEditors: Ethan
LastEditTime: 2026-08-02 16:55:06
Description: 
FilePath: /trainer/reward_model.py
'''


from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-1_8b-reward", trust_remote_code=True)

print(f"tokenizer: ", tokenizer)