# traditional-chinese-reading-comprehension-test-for-llms-312511050
traditional-chinese-reading-comprehension-test-for-llms-312511050 created by GitHub Classroom


檔案說明
---
AI.xlsx -train data   
AI1000.xlsx -test data   
xlsx_to_json.py -將xlsx檔的train data轉成json檔   
xlsx_to_json_test.py -將xlsx檔的test data轉成json檔   
AI.json -利用xlsx_to_json.py轉成json檔的train data   
AI1000.json --利用xlsx_to_json_test.py轉成json檔的test data   

---

## 設定環境
python=3.10.14   
pytorch=2.2.1   
unsloth=2024.5   
xformers=0.0.25.post1   
trl=0.8.6   
peft=0.11.1    
accelerate=0.30.1   
bitsandbytes=0.43.1

## 資料準備
1. 利用xlsx_to_json.py將AI.xlsx轉成訓練模型所需要的格式
```py
import pandas as pd
import json

# 讀取Excel檔
excel_file = '/home/4TB_storage_2/chieh_storage/DL/HW2/AI1000.xlsx'  # 請將 'your_excel_file.xlsx' 替換為實際的檔案路徑
df = pd.read_excel(excel_file, engine='openpyxl')

# 將每一行轉換為JSON格式
output_json_list = []
for index, row in df.iterrows():
    question = {
        "instruction": str(row["文章"]) + "\n",
        "input": "問題：" + str(row["問題"]) + "\n  1：" + str(row["選項1"]) + "\n  2：" + str(row["選項2"]) + "\n  3：" + str(row["選項3"]) + "\n  4：" + str(row["選項4"]) + "\n",
    }
    output_json_list.append(question)

# 將整體資料轉換為JSON格式
json_data = json.dumps(output_json_list, ensure_ascii=False, indent=2)

# 將JSON寫入檔案
json_file = 'AI1000.json'  # 請將 'output.json' 替換為希望輸出的JSON檔案路徑
with open(json_file, 'w', encoding='utf-8') as f:
    f.write(json_data)

print(f'成功將 {excel_file} 轉換為 {json_file}')

```
2. 利用xlsx_to_json_test.py將AI1000.xlsx轉成需模型預測資料的格式
```py

import pandas as pd
import json

# 讀取Excel檔
excel_file = '/home/4TB_storage_2/chieh_storage/DL/HW2/AI1000.xlsx'  # 請將 'your_excel_file.xlsx' 替換為實際的檔案路徑
df = pd.read_excel(excel_file, engine='openpyxl')

# 將每一行轉換為JSON格式
output_json_list = []
for index, row in df.iterrows():

    question = {
        "id": str(row["題號"]) + "\n",
        "instruction": str(row["文章"]) + "\n",
        "input": "問題：" + str(row["問題"]) + "\n  1：" + str(row["選項1"]) + "\n  2：" + str(row["選項2"]) + "\n  3：" + str(row["選項3"]) + "\n  4：" + str(row["選項4"]) + "\n",
        #"output": str(row["正確答案"])
    }
    output_json_list.append(question)

# 將整體資料轉換為JSON格式
json_data = json.dumps(output_json_list, ensure_ascii=False, indent=2)

# 將JSON寫入檔案
json_file = 'AI1000.json'  # 請將 'output.json' 替換為希望輸出的JSON檔案路徑
with open(json_file, 'w', encoding='utf-8') as f:
    f.write(json_data)

print(f'成功將 {excel_file} 轉換為 {json_file}')
```

## 訓練模型
依照LLM.ipynb內程式的順序執行。   
可以在以下程式更改想要使用的模型：
```py
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```
## 結果比較
|   | llama-3-8b-bnb-4bit |  gemma-7b-it-bnb-4bit |
|:-:|:-:|:-:|
|所需時間| 3.5hrs |  3hrs|
|epoch | 2| 2|
|batch_size| 8 |8|
|kaggle分數| 0.86000 |0.87571|
