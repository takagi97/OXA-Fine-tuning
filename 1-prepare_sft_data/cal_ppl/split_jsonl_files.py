import os

def split_jsonl_file(input_file, output_dir, lines_per_file=88):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_counter = 0
    current_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            current_lines.append(line)

            if line_number % lines_per_file == 0:
                output_file = os.path.join(output_dir, f"part_{file_counter:04d}.jsonl")
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(current_lines)
                file_counter += 1
                current_lines = []
        
        if current_lines:
            output_file = os.path.join(output_dir, f"part_{file_counter:04d}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.writelines(current_lines)

if __name__ == "__main__":
    input_file = "/path/to/jsonl"
    output_dir = "/path/to/splited"
    split_jsonl_file(input_file, output_dir, lines_per_file=33605)
