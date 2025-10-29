Welcome to RadCount by ehtbanton.

Current features:
  1) Use a browser UI via a Django server running at localhost:8000
  2) Start and interact with local LLM servers via llama.cpp
  3) Complete control over input context for one-shot LLM queries
  4) View CSV files for radiology report data (assumes 1 entry per row of CSV, with the top row containing field names)
  5) Custom extraction functions to create (smaller) LLM context files based on the CSV. Debug terminal in UI.

Setup instructions:
  1) Only run on a server with at least 8gb of graphics memory - otherwise, most local LLMs won't function. Tested on NVIDIA 3090 - should work on any NVIDIA gpu, or Apple silicon. 
  2) Clone the repo
  3) Install Python globally. Add directories to python and pip to the server's system environment variables.
  4) Run startup.py ("python startup.py" on Windows or "python3 startup.py" on Unix systems). This starts the server. If the first time running, it will create a venv, install all required packages, install llama.cpp, and download an example vision LLM - and then will start the server.
