import os
import json

def convert_jsonl_to_unicode_in_place():
    current_dir = os.getcwd()

    for filename in os.listdir(current_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(current_dir, filename)
            print(f"Checking {file_path}...")

            with open(file_path, 'r', encoding='utf-8') as infile:
                original_lines = infile.readlines()

            converted_lines = []
            changed = False

            for line_number, line in enumerate(original_lines, start=1):
                obj = json.loads(line)
                # if "src_text" is in obj, remove it
                if "src_text" in obj:
                    del obj["src_text"]
                # same for "prompt_instruction"
                if "prompt_instruction" in obj:
                    del obj["prompt_instruction"]
                # same for input
                if "input" in obj:
                    del obj["input"]
                # if hypothesis is a list, check it has only one element and take that element
                if isinstance(obj.get("hypothesis"), list) and len(obj["hypothesis"]) == 1:
                    obj["hypothesis"] = obj["hypothesis"][0]
                unicode_line = json.dumps(obj, ensure_ascii=False)
                converted_lines.append(unicode_line + '\n')

                if unicode_line.strip() != line.strip():
                    changed = True


            if changed:
                print(f"Overwriting {filename} with Unicode (ensure_ascii=False)...")
                with open(file_path, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(converted_lines)
                    
if __name__ == "__main__":
    convert_jsonl_to_unicode_in_place()
