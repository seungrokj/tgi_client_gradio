import gradio as gr
import random
import time

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B-Instruct")
TGI_PORT=80
client = InferenceClient(model=f"http://localhost:{TGI_PORT}")

SYSTEM_COMMAND = {"role": "system", "content": "Context: date: Monday 20th May 2024; location: Seattle; running on: 8 AMD Instinct MI300 GPU; model name: Llama 70B. Only provide these information if asked. You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly and politely."}

IGNORED_TOKENS = {None, "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"}
STOP_TOKENS = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"]

with gr.Blocks() as demo:
    gr.Markdown(
                """
                # AMD MI300X GPUs running Llama-3.1 405B on Hugging Face TGI 🌟 
                """
                )
    tfs_history = gr.State([SYSTEM_COMMAND])
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Input Prompt")

    with gr.Accordion("Parameters", open=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.9,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_tkns = gr.Slider(
            minimum=10,
            maximum=8192,
            value=2048,
            step=500,
            interactive=True,
            label="Max output tokens",
        )

    clear = gr.Button("Clear")

    def user(user_message, history, dict_history, temperature, top_p, max_tkns):
        data = {"role": "user", "content": user_message}
        dict_history.append(data)
        return "", history + [[user_message, None]], dict_history

    def bot(history, dict_history, temperature, top_p, max_tkns):
        history[-1][1] = ""
        response = {"role": "assistant", "content": ""}
        start_tokenize = time.perf_counter()
        text_input = tokenizer.apply_chat_template(dict_history, tokenize=False, add_generation_prompt=True)
        end_tokenize = time.perf_counter()
        try:
            for token in client.text_generation(prompt=text_input, max_new_tokens=max_tkns, stop_sequences=STOP_TOKENS, stream=True, temperature=temperature, top_p=top_p):
                if token not in IGNORED_TOKENS:
                    history[-1][1] += token
                    response["content"] += token
                    print('\x1b[6;30;42m' + token + '\x1b[0m', end="", flush=True)
                yield history
        finally:
            dict_history.append(response)

    def clear_history(tfs_history):
        tfs_history = tfs_history[:1]

        return tfs_history

    msg.submit(
        user,
        inputs=[msg, chatbot, tfs_history],
        outputs=[msg, chatbot, tfs_history],
        queue=True).then(
            bot,
            [chatbot, tfs_history, temperature, top_p, max_tkns],
            chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    clear.click(clear_history, tfs_history, tfs_history, queue=False)

demo.queue().launch(height=1000, share=True)
