# Note this file executes shell commands to download the MAESTRO dataset.
import subprocess

subprocess.run("wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
subprocess.run(["unzip maestro-v3.0.0-midi.zip", "-d", "maestro"])

# Delete unnecessary files, so the MAESTRO folder contains only MIDI files and can be iterated through successfully to create the dataset.
subprocess.run(["rm", "./maestro/maestro-v3.0.0/LICENSE"])
subprocess.run(["rm", "./maestro/maestro-v3.0.0/README"])
subprocess.run(["rm", "./maestro/maestro-v3.0.0/maestro-v3.0.0.csv"])
subprocess.run(["rm", "./maestro/maestro-v3.0.0/maestro-v3.0.0.json"])
