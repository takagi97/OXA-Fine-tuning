import os

def split_jsonl_file(input_file, output_dir, lines_per_file=88):
    # 如果输出目录不存在，则创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_counter = 0
    current_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            current_lines.append(line)
            # 每读取到lines_per_file行后就写入一个新文件
            if line_number % lines_per_file == 0:
                output_file = os.path.join(output_dir, f"part_{file_counter:04d}.jsonl")
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(current_lines)
                file_counter += 1
                current_lines = []
        
        # 如果剩余的行数不足lines_per_file，也写入一个新文件
        if current_lines:
            output_file = os.path.join(output_dir, f"part_{file_counter:04d}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                outfile.writelines(current_lines)

if __name__ == "__main__":
    input_file = "/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/jerrymu/SFT_for_RL/data/AceReason/AceReason-1.1-Math.uuid.gt.verify.true.shuf.jsonl"  # 请替换为你的大文件路径
    output_dir = "/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/jerrymu/SFT_for_RL/data/AceReason/AceReason-1.1-Math.uuid.gt.verify.true.shuf"         # 请替换为你期望输出的目录
    split_jsonl_file(input_file, output_dir, lines_per_file=33605)
