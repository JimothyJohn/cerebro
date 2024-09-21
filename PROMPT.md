# Prompt

## Problem

It's extremely costly and cumbersome to utilize vision perception models for niche use cases at the edge. I hope to make this easier, faster, cheaper, and more accessible by offering an efficient model accelerated in a cloud deployment. There's a deeper problem here, but this is the technical one.

## Goal

Create a Replicate A100 endpoint for a CUDA-accelerated version of Microsoft's multi-modal Phi-3 model, currently hosted on Hugging Face at [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct).

## Requirements

- The program should be easy to follow and check for potential errors.
- The program will provide detailed logging to help troubleshoot during development.
- When providing code, provide the entire program file, not just snippets.
- Make it clear in all files that they are AI-generated so the user knows to check for errors.
- Make it as generalizable as possible in case I want to deploy other models or methods.
- Provide suggestions for additional functionality.
- Keep all of the program explanations inside the files themselves as comments.
- Include error handling whenever possible.
- Download the model weights during image creation inside the `cog.yaml` file, not during the setup at program runtime.

## Outputs

- **PROMPT.txt**: Save this prompt in a more readable and easily applicable format as a plaintext file.
- **devcontainer.json**: A fully-featured development container file for VS Code that makes development instantaneous and lends itself well to GitHub Codespaces for cloud development.
- **.gitignore**: A .gitignore file suitable for this type of deployment.
- **tests.py**: Basic unit tests built using the pytest framework.
- **index.html** and **style.css**: An HTML version of the README that can be used in conjunction with GitHub Pages.
- **README.md**: A beautiful markdown file that elegantly explains what this repository does.
- **requirements.txt**: A file of the library dependencies.
- **cog.yaml**: A `cog.yaml` file that will serve as the configuration file for the container. The `cog.yaml` should include run instructions that download the model into the container image so it doesn’t have to be downloaded by `predict.py`.
- **predict.py**: A `predict.py` program that functions as the main inference program.

## Examples

Sample code for inference can be found at the [Phi-3.5-vision-instruct model page](https://huggingface.co/microsoft/Phi-3.5-vision-instruct#loading-the-model-locally).
