# Text-To-CLI-Model-and-Python
A Repository allowing for an AI model to generate and directly execute Linux Bash commands in an SSH terminal.

This project is powered by Python and Ollama \\add further notes here, python libs, etc...//

This project was initally prepaired and developed on a Jetson Orin Nano, future plans include extended training on an RTX 4070.  
<br>

The based model used for this project is the Qwen/Qwen2.5-Coder-1.5B-Instruct pulled from huggingface. 
(https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)

The dataset used to initally train the model was aelhalili/bash-commands-dataset. 
(https://huggingface.co/datasets/aelhalili/bash-commands-dataset)

<br>
<h3> Ollama and Model Installation Guide - this tutorial is for debian style distros of linux: </h3>

1. Download the unmerged gguf file - [Model/Unmerged/qwen2.5-cli-1.5b.gguf](https://github.com/billjoe0459/Text-To-CLI-Model-and-Python/blob/main/Model/Unmerged/qwen2.5-cli-1.5b.gguf)
2. Download the Modelfile - [Model/Unmerged/Modelfile](https://github.com/billjoe0459/Text-To-CLI-Model-and-Python/blob/main/Model/Unmerged/Modelfile)
3. Open VScode
4. Use standard SSH via VScode remote connections
5. Open work directory on your Jetson or linux computer from vscode
6. Install Ollama, run this in the ssh terminal - curl -fsSL https://ollama.com/install.sh | sh
7. Run the command and wait for install - ollama create cli-assistant -f Modelfile
8. Run the command - ollama run cli-assistant
9. Now you have the model, please have fun and be creative!
<br>

<h3>RELEASE NOTES:</h3> <br>
v1: trained qlora gguf of the model (you will need the preconfigued model file to download both the base model and cli model for use in ollama), only text to command works, future versions will include both a full gguf and python ssh support.

v2: created fp16 and Q4_K_M gguf files, loadable with standard model file/without needing to pull base model with ollama.
