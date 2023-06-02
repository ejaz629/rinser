# RINSER

We present a fast and accurate Windows API prediction system called RINSER: Accurate API Name prediction using transfer learning. 

## How to run?
There are two stages to run the RINSER properly:
- **Build API codeprints** There are two steps: first place binaries in the "binaries" folder and generate disassembly by running "start.py". Then run "rinser_from_disassembly.py". It will create API-codeprints in the folder "api-codeprints" and their normalized form which are placed in "api-codeprints-for-llm" folder. The normalized API-codeprints are used against RINSER-LLM in the next step to predict API names.
- **Predict API names using RINSER-LLM** The generated API-codeprints are used to test against RINSER-LLM (placed in "RINSER-LLM" folder) using "predict.py" file. The test dataset sample is already placed in these folders. Note that the "binaries" folder contain malicous binaries in compressed (NOT password-protected) format. Be careful!

The dependencies are given in requirement file. If you just want to test (not building API codeprints from your own set of binaries), you need to install dependencies in (2) of the requirements.txt file.
