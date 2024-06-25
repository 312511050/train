# traditional-chinese-tv-show-caption-recognition-using-lmms-312511050
traditional-chinese-tv-show-caption-recognition-using-lmms-312511050 created by GitHub Classroom


檔案說明
---
synth.py - 將字幕貼到圖片上   
train_sub - 從新兵日記及時尚玩家的yt影片上獲得的cc字幕，執行synth.py所需的字幕檔   
fit_video_chinese_font - 執行synth.py所需的字體檔案   
image - 從ImagetNet下載作為dataset，為字幕的背景圖(未包含字幕)   
out.txt - 執行完synth.py所產生的檔案，內容為每一張照片對應的字幕   
train_data - 執行完synth.py所產生的圖片檔(包含字幕)   
convert.py - 將train_data及out.txt轉換成訓練所需的資料   
train.json - 訓練所需的資料   
test.py - 產出結果

---

## 環境
1. 下載Tinyllava GitHub
```
git clone https://github.com/TinyLLaVA/TinyLLaVA_Factory.git
cd TinyLLaVA_Factory
```
2. 安裝所需套件
```
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

## 資料準備
1. 至imagenet下載圖片，作為字幕背景圖
   https://www.image-net.org/download.php
2. 至youtube尋找新兵日記及時尚玩家有cc字幕的影片，並複製下來到train_sub
3. 利用synth.py合成訓練所需圖片(將train_sub的字幕合成至圖片上)
   ```py
    import os
    import random
    from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageEnhance
    import time
    
    def generate_synthetic_data(output_dir, num_images, text_file, image_dir,image_file):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Read sentences from text file
        with open(text_file, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file.readlines()]
    
        # Get list of image files from the image directory
        # image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
        # Get list of font files from the font directory
        for i in range(num_images):
            print(i )
            
            # Randomly select an image and font
            selected_image = Image.open(os.path.join(image_dir, random.choice(image_file)))
            
            width, height = selected_image.size
    
            if (height<100 or height*width>2200000):
                continue
            
            area= height*width
            if 1560800<area<=2200000:
                selected_image=selected_image.resize((1920,1080))
                width, height =1920,1080
                font_size=(75,100)
            elif 660000<area<=1560800:
                selected_image=selected_image.resize((1280,720))
                width, height =1280,720
                font_size=(50,70)
            elif 315000<area<=660000:
                selected_image=selected_image.resize((854,480))
                width, height =854,480
                font_size=(35,48)
            elif area<=315000:
                selected_image=selected_image.resize((640,360))
                width, height =640,360
                font_size=(27,36)
            else:
                selected_image=selected_image.resize((640,360))
                width, height =640,360
                font_size=(27,36)
    
    
    
            # 如果是灰度圖像，將其轉換為 RGB 格式
            selected_image = selected_image.convert("RGB")
            
            # Create a blank image with the same size as the selected image
            new_image = selected_image.copy()
            draw = ImageDraw.Draw(new_image)
    
    
    
            # Calculate text position horizontal
            def create_text(text_height,sentence,text_real_width,text_real_height,text_color,selected_font):
    
                text_position = ((width-text_real_width)//2, text_height )
                region_right =text_position[0] +text_real_width
                draw.text(text_position, sentence, font=selected_font, fill=text_color)
                    
                region_left = text_position[0]
                region_top = text_position[1]
                region_bottom =  text_position[1] +text_real_height
    
                # Find maximum brightness within the region
                max_brightness = 0
                for x in range(region_left, region_right + 1):
                    for y in range(region_top, region_bottom + 1):
                        try:
                            pixel_brightness = sum(selected_image.getpixel((x, y))) // 3  # Calculate brightness (average of RGB)
                        except:
                            continue
                        max_brightness = max(max_brightness, pixel_brightness)
    
                # Check if maximum brightness exceeds threshold
                brightness_threshold = 150  # Adjust this threshold as needed
                if max_brightness > brightness_threshold:
                    border_thickness = 1
                    for dx in range(-border_thickness, border_thickness + 1):
                        for dy in range(-border_thickness, border_thickness + 1):
                            draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0,0,0))
                    draw.text(text_position, sentence, font=selected_font, fill=text_color)
                else:
                    border_thickness = 1
                    if random.randint(0,3)==2:
                        thick=random.randint(1,3)
                        for dx in range(-border_thickness, border_thickness + thick):
                            for dy in range(-border_thickness, border_thickness + thick):
                                draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0,0,0))
                    draw.text(text_position, sentence, font=selected_font, fill=text_color)
    # end of def()
    
    
            gt_string=""
            random.seed()
            num=0
            out_range=1
    
            text_color=(random.randint(250,255),random.randint(250,255),random.randint(250,255))
            font_dir="fit_video_chinese_font"
            font_files = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
            while out_range==1:
                sentence=random.choice(sentences)
                sentence=sentence.replace("\n","")
                size=random.randint(font_size[0],font_size[1])
                selected_font = ImageFont.truetype(os.path.join(font_dir, random.choice(font_files)), size)
                text_bbox = draw.textbbox((0,0), sentence, font=selected_font)
                text_real_width = text_bbox[2] - text_bbox[0]
                text_real_height = text_bbox[3] - text_bbox[1]
                print(text_real_height,text_real_width)
                #text_real_width, text_real_height = draw.textsize(sentence, font=selected_font)
                line1=height-int(text_real_height*3)
                line2=height-(text_real_height*2)
                if width-text_real_width>=0:
                    
                    create_text(random.randint(line1,line2),sentence,text_real_width,text_real_height,text_color,selected_font)
                    out_range=0
                else:
                    num=num+1
                    print("第{}張有一行跳過".format(i))
                    with open ("mistake.txt","a",encoding="utf-8") as fw:
                        fw.write("第{}張有一行跳過".format(i)+"\n")
                if num ==10:
                    print(num)       
                    break
            if num ==10:
                continue
            gt_string=gt_string+sentence+" "
    
           
                
    
            output_path = os.path.join("image_{}.jpg".format(i + 1))
            with open ("out.txt","a",encoding="utf-8") as f:
                f.write(output_path+" "+gt_string+"\n")
            
    
            #Image.blend(selected_image, new_image, alpha=0.5).save(output_path)
            alpha_value = 255  # You can adjust this value
            Image.blend(selected_image, new_image, alpha=alpha_value/255.0).save(
                os.path.join(output_dir, "image_{}.jpg".format(i + 1))
            )
    
    
    
    
    if __name__ == "__main__":
        output_directory = 'train_data'
        number_of_images = 5000
        text_file_path = "train_sub"
        image_directory = "image"
        image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
        
        #text_color = (255, 255, 255)  # RGB color
    
        generate_synthetic_data(output_directory, number_of_images, text_file_path, image_directory,image_files)
   ```
4. 利用convert.py將synth.ph所產生上字幕的圖片及每張圖片對應的字幕的檔案(out.txt)轉換成訓練模型所需要的資料
   ```py
    import shortuuid
    import os
    import json
    import random
    import tqdm
    from PIL import Image
    
    # Define paths
    image_folder_path = '/home/disk/chieh/DL/HW3/synth/train_data'
    text_file_path = '/home/disk/chieh/DL/HW3/synth/out.txt'
    output_json_path = '/home/disk/chieh/DL/HW3/synth/train.json'
    
    # Load and parse text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Prepare description list
    description_list = [
        "Identify and describe subtitles present in the image.",
        "Locate and provide the text of subtitles within the picture.",
        "Extract and report subtitles found in the image.",
        "Detect and describe the content of subtitles in the picture.",
        "Identify the text of subtitles in the image provided."
    ]
    
    data = []
    
    # Process each line in the text file
    for line in tqdm.tqdm(lines):
        # Split the line to get the image filename and subtitle
        parts = line.strip().split(' ', 1)
        if len(parts) < 2:
            continue
        image_filename, subtitle = parts
    
        # Generate a unique ID
        uuid = shortuuid.uuid()
    
        # Create the sample dictionary
        sample_dict = {
            "id": uuid,
            "image": os.path.join(uuid + '.jpg'),
            "conversations": [
                {"from": "human", "value": "<image>\n" + random.choice(description_list)},
                {"from": "gpt", "value": subtitle}
            ]
        }
    
        # Save the image with the new UUID filename
        original_image_path = os.path.join(image_folder_path, image_filename)
        new_image_path = os.path.join(image_folder_path, uuid + '.jpg')
        image = Image.open(original_image_path)
        image.save(new_image_path)
    
        # Add the sample to the dataset
        data.append(sample_dict)
    
    # Save the dataset to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

   
   ```

## 訓練finetune模型
在下載的GitHub中有一個custom_finetune.sh(TinyLLaVA_Factory/scripts/train/custom_finetune.sh)   
將路徑改成自已所需要的路徑，並在TinyLLaVA_Factory資料夾底下執行
```sh
bash scripts/train/custom_finetune.sh
```
```sh
DATA_PATH="/home/disk/chieh/DL/HW3/TinyLLaVA_Factory/dataset/train.json"
IMAGE_PATH="/home/disk/chieh/DL/HW3/TinyLLaVA_Factory/dataset/train_data"
MODEL_MAX_LENGTH=512
OUTPUT_DIR="/home/disk/chieh/DL/HW3/TinyLLaVA_Factory/checkpoints/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"
ZERO="/home/disk/chieh/DL/HW3/TinyLLaVA_Factory/scripts/zero2.json"

deepspeed --include localhost:0 --master_port 29501 tinyllava/train/custom_finetune.py \
    --deepspeed $ZERO \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version phi \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 True \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora


```

## inference
利用test.py產生出test資料夾內圖片的字幕，並存入csv檔
```py
import os
import csv
from tinyllava.eval.run_tiny_llava import eval_model

# 定義資料夾路徑和其他參數
image_folder = "/home/disk/chieh/DL/HW3/test"
output_csv = "output.csv"
model_path = "/home/disk/chieh/DL/HW3/model.pth"
prompt = "Identify the text of subtitles in the image provided."
conv_mode = "phi" # or llama, gemma, etc

# 定義處理圖片並保存結果的函數
def process_images(image_folder, output_csv, model_path, prompt, conv_mode):
    # 獲取資料夾中的所有圖片路徑
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    # 準備寫入 CSV 的資料
    csv_data = [("id", "text")]

    for image_file in image_files:
        args = type('Args', (), {
            "model_path": model_path,
            "model": None,
            "query": prompt,
            "conv_mode": conv_mode,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        # 進行模型推理並獲取輸出
        output_text = eval_model(args)
        
        # 獲取圖片名稱作為 ID
        image_name = os.path.basename(image_file).split('.')[0]
        
        # 添加到 CSV 資料
        csv_data.append((image_name, output_text))

    # 將結果寫入 CSV 檔案
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

# 執行處理函數
process_images(image_folder, output_csv, model_path, prompt, conv_mode)

```

