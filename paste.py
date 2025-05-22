import os

output_file = "/home/ubuntu/all_code.txt"

with open(output_file, "w") as outfile:
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                outfile.write(f"===== {filepath} =====\n")
                try:
                    with open(filepath, "r") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")
                outfile.write("\n\n")
