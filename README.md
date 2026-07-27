# Text-To-CLI-Model-and-Python
A Repository allowing for an AI model to generate and directly execute Linux Bash commands in an SSH terminal.

This project is powered by Python and Ollama \\add further notes here, python libs, etc...//

This project was initally prepaired and developed on a Jetson Orin Nano, future plans include extended training on an RTX 4070.  
<br>

The based model used for this project is the Qwen/Qwen2.5-Coder-1.5B-Instruct pulled from huggingface. <br>
(https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)

The dataset used to initally train the model was aelhalili/bash-commands-dataset. <br>
(https://huggingface.co/datasets/aelhalili/bash-commands-dataset)
<br>
<br>
The dataset used to train the unsloth variant was mecha-org/linux-command-dataset
<br>
(https://huggingface.co/datasets/mecha-org/linux-command-dataset)
<br>
<h3> Ollama and Model Installation Guide - this tutorial is for a server with a debian style distro of linux: </h3>

1. Download the unmerged unsloth gguf file - [/Unsloth_Model/Unmerged_Unsloth/qwen2.5-cli-unmerged.gguf](https://github.com/billjoe0459/Text-To-CLI-Model-and-Python/blob/main/Unsloth_Model/Unmerged_Unsloth/qwen2.5-cli-unmerged.gguf)
2. Download the Modelfile - [Unsloth_Model/Unmerged_Unsloth/Modelfile.unslothed](https://github.com/billjoe0459/Text-To-CLI-Model-and-Python/blob/main/Unsloth_Model/Unmerged_Unsloth/Modelfile.unslothed)
3. Open VScode
4. Use standard SSH via VScode remote connections
5. Open work directory on your Jetson or linux computer from vscode
6. Install Ollama, run this in the ssh terminal - curl -fsSL https://ollama.com/install.sh | sh
7. Run the command and wait for install - ollama create qwen2.5-cli:1.5b-q4_K_M -f Modelfile.unslothed
8. Run the command - ollama run qwen2.5-cli:1.5b-q4_K_M
9. Now you have the model, please have fun and be creative! <br> Here is a demo video: (https://youtu.be/edBHgl2VJtY) -- as of 7/26/26 this is out of date, will record a new demo when the time comes
<br>

<h3>RELEASE NOTES:</h3> <br>
v1: trained qlora gguf of the model (you will need the preconfigued model file to download both the base model and cli model for use in ollama), only text to command works, future versions will include both a full gguf and python ssh support

v2: created fp16 and Q4_K_M gguf files, loadable with standard model file/without needing to pull base model with ollama

v3: reorganized the the file system, included q4_k_m merged and unmerged unsloth models with appropriate model files