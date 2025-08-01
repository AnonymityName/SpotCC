import sys

def extract_ips(log_file_path):
    encode_ips = []
    preempted_ips = []
    decoder_times = 0
    recompute_times = 0
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if "Choose backend IP:" in line and "Encode id:" in line:
                ip = line.split("Choose backend IP:")[1].split()[0]
                encode_ips.append(ip)
            elif "new preempted nodes size:1" in line:
                if i + 1 < len(lines):
                    ip = lines[i + 1].strip()
                    preempted_ips.append(ip)
            elif "recompute!" in line:
                recompute_times+=1
            elif "decoder performed ~" in line:
                decoder_times+=1
                
    return encode_ips, preempted_ips


if __name__ == "__main__":
    log_file_path = sys.argv[1]
    encode_ips, preempted_ips = extract_ips(log_file_path)
    print(encode_ips)
    print(len(encode_ips))
    print(preempted_ips)