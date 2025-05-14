import os


def parse_enrollment_file(enrollment_paths):
    enroll_dict = {}
    for path in enrollment_paths:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                speaker_id = parts[0]
                utterances = parts[1].split(',')
                enroll_dict[speaker_id] = utterances
    return enroll_dict

def parse_trial_file(trial_path, enroll_dict):
    output_lines = []
    with open(trial_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            speaker_id = parts[0]
            test_utt_id = parts[1]
            key = parts[3]  # target/nontarget/spoof

            if speaker_id in enroll_dict:
                enrollment_utts = enroll_dict[speaker_id]
                enroll_str = "[" + ",".join(f'"{utt}"' for utt in enrollment_utts) + "]"
                output_line = f"{key}, {speaker_id}:{enroll_str}, {speaker_id}/{test_utt_id}"
                output_lines.append(output_line)
            else:
                print(f"[WARNING] Speaker ID {speaker_id} not found in enrollment data.")
    return output_lines

# eval files
female_trn_eval = '/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trn.txt'
male_trn_eval = '/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trn.txt'
gi_trl_eval = '/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt'

# dev files
female_trn_dev = '/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.female.trn.txt'
male_trn_dev = '/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.male.trn.txt'
gi_trl_dev = '/home/hansini/Campus/FYP/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt'


# Step 1: Parse enrollment files
enroll_data_eval = parse_enrollment_file([female_trn_eval, male_trn_eval])

# Step 2: Parse trial file and generate new verify-like structure
verify_like_data = parse_trial_file(gi_trl_eval, enroll_data_eval)

#if file exists, do not create again
if os.path.exists('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Eval/verify_eval_gi.txt'):
    print("File already exists. Exiting.")
    exit()
else:
    print("File does not exist. Creating new file.")
    # Step 3: Write to file
    with open('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Eval/verify_eval_gi.txt', 'w') as f:
        for line in verify_like_data:
            f.write(line + '\n')

    print("Verify eval file created successfully.")

enroll_data_dev = parse_enrollment_file([female_trn_dev,male_trn_dev])
verify_like_data_dev = parse_trial_file(gi_trl_dev, enroll_data_dev)

# Check if the file already exists
if os.path.exists('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Eval/verify_dev_gi.txt'):
    print("File already exists. Exiting.")
    exit()
else:
    print("File does not exist. Creating new file.")
    # Step 3: Write to file
    with open('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Eval/verify_dev_gi.txt', 'w') as f:
        for line in verify_like_data_dev:
            f.write(line + '\n')
        print("Verify dev file created successfully.")




