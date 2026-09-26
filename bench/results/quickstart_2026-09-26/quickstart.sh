set -euo pipefail
M=Qwen/Qwen2.5-0.5B-Instruct
entropy keygen --out master.key
entropy seal --model $M --key master.key --max-new-tokens 400 \
  --prompt "Write a short story about a lighthouse keeper who finds a letter in a bottle." --out sealed.json
python3 -c 'import json; d = json.load(open("sealed.json")); open("candidate.txt", "w").write(d["text"])'
TOKEN=$(python3 -c 'import json; print(json.load(open("sealed.json"))["token"])')
entropy verify --model $M --text-file candidate.txt --token "$TOKEN" --key master.key
entropy detect --model $M --text-file candidate.txt --key master.key
