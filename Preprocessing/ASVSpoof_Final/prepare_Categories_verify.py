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

# VC files
female_trn_vc = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/VC/vc_enrol_female.txt'
male_trn_vc = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/VC/vc_enrol_male.txt'
gi_trl_vc = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/VC/vc.txt'

# TTS files
female_trn_tts = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/TTS/tts_enrol_female.txt'
male_trn_tts = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/TTS/tts_enrol_male.txt'
gi_trl_tts = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/TTS/tts.txt'

# Bonafide files
female_trn_sasv = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/Bonafide/bonafide_enrol_female.txt'
male_trn_sasv = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/Bonafide/bonafide_enrol_male.txt'
gi_trl_sasv = '/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/Bonafide/bonafide.txt'



enroll_data_vc = parse_enrollment_file([female_trn_vc, male_trn_vc])
verify_like_data_vc = parse_trial_file(gi_trl_vc, enroll_data_vc)

#if file exists, do not create again
if os.path.exists('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_vc_eval_gi.txt'):
    print("File already exists. Exiting.")
    exit()
else:
    print("File does not exist. Creating new file.")
    # Step 3: Write to file
    with open('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_vc_eval_gi.txt', 'w') as f:
        for line in verify_like_data_vc:
            f.write(line + '\n')

    print("Verify vc eval file created successfully.")



enroll_data_tts = parse_enrollment_file([female_trn_tts,male_trn_tts])
verify_like_data_tts = parse_trial_file(gi_trl_tts, enroll_data_tts)


# Check if the file already exists
if os.path.exists('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_tts_eval_gi.txt'):
    print("File already exists. Exiting.")
    exit()
else:
    print("File does not exist. Creating new file.")
    # Step 3: Write to file
    with open('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_tts_eval_gi.txt', 'w') as f:
        for line in verify_like_data_tts:
            f.write(line + '\n')
        print("Verify tts eval file created successfully.")



enroll_data_sasv = parse_enrollment_file([female_trn_sasv,male_trn_sasv])
verify_like_data_sasv = parse_trial_file(gi_trl_sasv, enroll_data_sasv)

# Check if the file already exists
if os.path.exists('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_bonafide_eval_gi.txt'):
    print("File already exists. Exiting.")
    exit()
else:
    print("File does not exist. Creating new file.")
    # Step 3: Write to file
    with open('/home/hansini/Campus/FYP/SonicCypher/Preprocessing/ASVSpoof_Final/verify_bonafide_eval_gi.txt', 'w') as f:
        for line in verify_like_data_sasv:
            f.write(line + '\n')
        print("Verify bonafide eval file created successfully.")
