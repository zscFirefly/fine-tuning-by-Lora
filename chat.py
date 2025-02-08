import streamlit as st
import torch

st.set_page_config(page_title="LLaMA3 微调", page_icon="🤖", )
hide_streamlit_style = """
<style>#root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 1rem;}</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("💬 LLaMA3-8B 微调演示（甄嬛版）")

# 检查CUDA是否可用，然后检查MPS是否可用，最后回退到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_id = 'LLM-Research/Meta-Llama-3-8B-Instruct'
models_dir = './models'
model_path = f"{models_dir}/model/{model_id.replace('.', '___')}"
lora_dir = f"./models/lora/{model_id}"
torch_dtype = torch.bfloat16


# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def init_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from peft import PeftModel
    from peft import LoraConfig, TaskType

    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch_dtype)
    # 加载lora权重

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = PeftModel.from_pretrained(model, model_id=lora_dir, config=config)
    return tokenizer, model


def bulid_input(prompt, history=[]):
    system_format = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format = '<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n'
    history.append({'role': 'user', 'content': prompt})
    prompt_str = ''
    # 拼接历史对话
    for item in history:
        if item['role'] == 'user':
            prompt_str += user_format.format(content=item['content'])
        else:
            prompt_str += assistant_format.format(content=item['content'])
    return prompt_str + '<|start_header_id|>assistant<|end_header_id|>\n\n'


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 加载LLaMA3的model和tokenizer
tokenizer, model = init_model()

if prompt := st.chat_input(placeholder="我是你的环环，咱俩聊点啥吧~"):
    st.chat_message("user").write(prompt)
    print(f"\n### 用户输入：{prompt} ###\n")

    # 构建输入
    input_str = bulid_input(prompt=prompt, history=st.session_state["messages"])
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = (response.strip()
                .replace('<|eot_id|>', "")
                .replace('<|start_header_id|>assistant<|end_header_id|>', '')
                .replace('<|end_of_text|>', '').strip())

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    print(f"\n### 模型回复：{response} ###\n")
    print(st.session_state)
