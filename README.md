# RINSER

We present a fast and accurate Windows API prediction system called RINSER: Accurate API Name prediction using transfer learning. 

## How to run?
There are two stages to run the RINSER properly:
- **Build API codeprints** There are two steps: first place binaries in the "binaries" folder and generate disassembly by running "start.py". Then run "rinser_from_disassembly.py". It will create API-codeprints in the folder "api-codeprints" and their normalized form which are placed in "api-codeprints-for-llm" folder. The normalized API-codeprints are used against RINSER-LLM in the next step to predict API names.
- **Predict API names using RINSER-LLM** The generated API-codeprints are used to test against RINSER-LLM (placed in "RINSER-LLM" folder) using "predict.py" file. The test dataset sample is already placed in these folders. Note that the "binaries" folder contain malicous binaries in compressed (NOT password-protected) format. Be careful!

### Prediction with the RINSER's pretrained model
- The dependencies are given in requirement.txt file. If you just want to test (not building API codeprints from your own set of binaries), you need to only install dependencies in (3) of the requirements.txt file.

- If you want to extract API-codeprints from your own dataset, place all binaries in the "binaries" folder and follow the steps mentioned above in **Build API codeprints**. However, for the scripts to run, you need to have respective dependencies mentioned in (2) of the requirement.txt file.

## Setup the environment
1. First you need to add IDA-Pro path in the Windows environment variable in to use **idat64** tool for disassembling the binaries.

2. Required packages for extracting **API-codeprints**:
conda
pandas
tqdm
nltk

3. Required for building and using RINSER-LLM model for inference.
transformers
tokenizers version 0.13.3
Xformers
pre-trained LLM
  -In addition, you need rinser-llm pretrained model to be available in the main folder.
