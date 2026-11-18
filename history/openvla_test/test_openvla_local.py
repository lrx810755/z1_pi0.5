import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

model_path = "/home/lp531/models/openvla-7b"

print("Loading processor from local path...")
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

print("Loading model from local path...")
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
).to("cuda:0")

print("Model loaded successfully.")

# 先用一张纯白测试图验证流程能否跑通
image = Image.new("RGB", (224, 224), color=(255, 255, 255))

instruction = "pick up the red block"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

print("Running inference...")
action = vla.predict_action(
    **inputs,
    unnorm_key="bridge_orig",
    do_sample=False,
)

print("Predicted action:")
print(action)
