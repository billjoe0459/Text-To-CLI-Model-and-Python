# Text-To-CLI-Model-and-Python
A Repository allowing for an AI model to generate and directly execute Linux Bash commands in an SSH terminal.

This project is powered by Python and Ollama \\add further notes here, python libs, etc...//

This project was initally prepaired and developed on a Jetson Orin Nano, future plans include extended training on an RTX 4070.  
<br>

The based model used for this project is the Qwen/Qwen2.5-Coder-1.5B-Instruct pulled from huggingface.

The dataset used to initally train the model was aelhalili/bash-commands-dataset.

<h3>RELEASE NOTES:</h3> <br>
v1: trained qlora gguf of the model (you will need the preconfigued model file to download both the base model and cli model for use in ollama), only text to command works, future versions will include both a full gguf and python ssh support.

v2: created fp16 and Q4_K_M gguf files, loadable with standard model file/without needing to pull base model with ollama.