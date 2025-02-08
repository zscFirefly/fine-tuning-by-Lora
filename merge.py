from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model


model_id = 'qwen/Qwen1.5-4B-Chat'
device = "cpu"
models_dir = './models'
model_path = f"{models_dir}/model/{model_id.replace('.', '___')}"
checkpoint_dir = f"./models/checkpoint/{model_id}"
lora_path = f"./models/lora/{model_id}"
# 合并后的模型路径
output_path = f"./models/output/{model_id}"


def merge_model():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 等于训练时的config参数
    config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
      inference_mode=False,  # 训练模式
      r=8,  # Lora 秩
      lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
      lora_dropout=0.1  # Dropout 比例
    )


    # base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cpu")
    # base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    lora_model = PeftModel.from_pretrained(base, model_id=lora_path, config=config)
    # lora_model = PeftModel.from_pretrained(
    #     base,
    #     lora_path,
    #     torch_dtype=torch.float16,
    #     config=config
    # )
    model = lora_model.merge_and_unload()
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


def chat():
    device = "cpu"
    model_name_or_path = output_path

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cpu", torch_dtype="auto")
    # 准备输入提示
    prompt = "你是谁？"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成文本
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_length=50,
        max_new_tokens=512,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0],
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=model_inputs.attention_mask,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response.strip())




if __name__ == '__main__':
    chat()


