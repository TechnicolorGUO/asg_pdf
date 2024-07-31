import subprocess

# 调用 nvidia-smi 命令
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
print(result.stdout.decode('utf-8'))



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import torch
import os

def generate(context, question, temp=0.7):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # server
    model_path = "/home/guest01/develope/web/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # colab
    # model.to(device)

    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=0 if torch.cuda.is_available() else -1,
    #     batch_size=1,
    #     max_new_tokens=200,
    #     num_beams=4,
    #     do_sample=True,
    #     top_p=0.8,
    #     temperature=temp,
    #     repetition_penalty=1.5
    # )

    template = """Use the following pieces of context to answer the question at the end.
    Provide a precise answer based on the context and attach the source coordinate SC of your answer in [SC]:
    ```
    ============================================================
    @3603a49a-d26c-4a2d-a9f1-0a608211b3ba//Verun is a current year-3 student of CS//
    @7628b67b-3eb1-437c-b302-f0df15847605//Verun is a boy//
    @fbb63230-714f-4b53-bfcc-20ca1f4ff734//Verun is from San Diego//
    END OF RESULT//
    ============================================================

    Question: Who is Verun?
    ============================================================
    ```
    Think and respond with [@SC] in several complete and logical sentences:
    ```
    THOUGHT: The question ~~Who is Verun?~~ is asking for Verun's information.

    ANSWER: Verun is a student from San Diego[SC: @fbb63230-714f-4b53-bfcc-20ca1f4ff734] and currently in CS[SC: @3603a49a-d26c-4a2d-a9f1-0a608211b3ba].

    THOUGHT: 'Verun is a student from San Diego[SC: @fbb63230-714f-4b53-bfcc-20ca1f4ff734]' is not related to '@7628b67b-3eb1-437c-b302-f0df15847605//HAER is a boy//'.

    FINAL ANSWER: As far as I know, Verun is a student from San Diego[SC: @e956dd] and currently in CS[SC: @0a3f01].
    ```

    Now do the real task below!

    ============================================================
    {context}
    ============================================================
    Question: {question}
    ============================================================

    """

    prompt_template = PromptTemplate.from_template(template)
    formatted_prompt = prompt_template.format(context=context, question=question)

    # generate answer
    inputs = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=1000, temperature=temp)
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # inputs = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    # outputs = model.generate(inputs, max_length=1000, temperature=temp)
    # res = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract final answer directly in the generate function
    final_answer_start = "FINAL ANSWER:"
    start_index = res.find(final_answer_start)
    if start_index != -1:
        final_answer = res[start_index + len(final_answer_start):].strip()
    else:
        final_answer = "No final answer found."

    return final_answer


context = """
@3603a49a-d26c-4a2d-a9f1-0a608211b3ba//Verun is a current year-3 student of CS//
@7628b67b-3eb1-437c-b302-f0df15847605//Verun is a boy//
@fbb63230-714f-4b53-bfcc-20ca1f4ff734//Verun is from San Diego//
END OF RESULT//
"""

question = "Who is Verun?"

result = generate(context, question)
print(result)
