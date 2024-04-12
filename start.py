import os
import glob
import subprocess

folder_path = r"binaries"


# Define the file extensions to consider
extensions = (".dll", ".exe")

# Get a list of file paths matching the specified extensions
file_paths = glob.glob(os.path.join(folder_path, "*.*"))


# Iterate over the file paths
print("\nDisassemby started ...")
for file_path in os.listdir(folder_path):
    _, file_extension = os.path.splitext(file_path)
    filename = os.path.basename(file_path)

    if file_extension:
        filename = filename.split('.')[0]
        
    if not os.path.splitext(filename)[1] or file_extension in extensions:
        binary = os.path.join(folder_path, filename)
        command = f'idat64 -c -A -Lida-logs\ida.txt -S"myscript.py disassembly\\{filename}" '+ binary

        print("\n")
        print(f'Disassembling the binary: {filename}')
        process = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ### Wait for the subprocess to finish
        stdout, stderr = process.communicate()

        ### Check the return code of the subprocess
        return_code = process.returncode

        ### Process the output or errors if needed
        ### ...
        
        ## Display completion message
        print("Disassembled the binary, and written to \"disassembly\" folder. Return code:", return_code)


