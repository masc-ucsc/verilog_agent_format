from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import StoppingCriteria, StoppingCriteriaList 
# Path to the trained model checkpoint
model_path = ""
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.float16,device_map="auto")

prompt = """<|im_start|>system
You are a Verilog simplifier. Convert Verilog code with temporary LiveHD wires into clean and readable Verilog.<|im_end|>
<|im_start|>user
module multi_stage_0(
   input [15:0] input_data_1001
  ,output reg [15:0] output_data_2002
);
reg [15:0] _stage1_temp;
reg [15:0] _stage2_temp;
reg [15:0] _stage3_temp;
reg [15:0] output_data_2002_u;
always_comb begin
  _stage1_temp = input_data_1001;
end
always_comb begin
  _stage2_temp = _stage1_temp;
end
always_comb begin
  _stage3_temp = _stage2_temp;
end
always_comb begin
  output_data_2002_u = _stage3_temp;
end
always_comb begin
  output_data_2002 = output_data_2002_u;
end
endmodule
<|im_end|>
<|im_start|>assistant
"""
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.stop_token_ids

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
token_count = len(inputs["input_ids"][0])
print(token_count)
tokenizer.eos_token = "<|im_end|>"
tokenizer.eos_token_id = 151645
model.config.eos_token_id = 151645
# Generate the output with formatting
outputs = model.generate(**inputs, max_new_tokens=1028,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=StoppingCriteriaList([
        StopOnTokens([tokenizer.convert_tokens_to_ids("<|im_start|>")])
    ]))
decoded = tokenizer.convert_ids_to_tokens(outputs[0])

# Decode and print the result
formatted_verilog = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(formatted_verilog)
