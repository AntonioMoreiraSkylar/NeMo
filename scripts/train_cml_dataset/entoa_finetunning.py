import os
import json
import subprocess
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from pydub import AudioSegment
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset

# This must be on DockerFile
# os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["OC_CAUSE"] = "1"

NEMO_DIR = Path('/home/antonio/NeMo/') #/usr/src/app/
DATA_DIR = Path('/home/antonio/data/') #/mnt/fs/data

#FASTPITCH_PATH = Path("/mnt/fs/data/cml-portuguese-2/exps/FastPitch/fast_run/checkpoints/FastPitch.nemo")
FASTPITCH_PATH = Path('/home/antonio/.cache/torch/NeMo/NeMo_2.1.0/tts_en_fastpitch_align/b7d086a07b5126c12d5077d9a641a38c/tts_en_fastpitch_align.nemo')
FINETUNING_SCRIPT_PATH = NEMO_DIR / 'examples' / 'tts' / 'fastpitch_finetune.py'
CONFIG_TTS_PATH = NEMO_DIR / 'examples' / 'tts' / 'conf' / 'fastpitch_align_v1.05.yaml'

TRAIN_DATASET_PATH = DATA_DIR / '9017_manifest_train_dur_5_mins_local.json'
VALIDATION_DATASET_PATH = DATA_DIR / '9017_manifest_dev_ns_all_local.json'

SUP_DATA_PATH = DATA_DIR / 'fastpitch_sup_data/'
EXP_MANAGER_DIR = DATA_DIR / 'ljspeech_to_9017_no_mixing_5_mins'

# CMU_DICT_PATH = NEMO_DIR / 'scripts' / 'tts_dataset_files' / 'ipa_cmudict-0.7b_nv23.01.txt'
CMU_DICT_PATH = NEMO_DIR / 'scripts' / 'tts_dataset_files' / 'cmudict-0.7b_nv22.10'
HETERONYMS_PATH = NEMO_DIR / 'scripts' / 'tts_dataset_files' / 'heteronyms-052722'

assert all([
    NEMO_DIR.exists(),
    DATA_DIR.exists(),
    FASTPITCH_PATH.exists(),
    FINETUNING_SCRIPT_PATH.exists(),
    CONFIG_TTS_PATH.exists(),
    TRAIN_DATASET_PATH.exists(),
    VALIDATION_DATASET_PATH.exists(),
    CMU_DICT_PATH.exists(),
    HETERONYMS_PATH.exists(),
])

# AUDIO_ROOT_PATH = Path('/mnt/fs/entoa/')
AUDIO_ROOT_PATH = Path("/home/antonio/entoa_prosodic/")
COMMON_SPEAKERS = {
    'SP_D2_255_TB-DOC1',
    'SP_D2_255_TB-DOC2',
    'SP_D2_255_TB-L1',
    'SP_D2_255_TB-L2',
    'SP_D2_333_TB-L1',
    'SP_D2_333_TB-L2',
    'SP_D2_343_TB-DOC1',
    'SP_D2_343_TB-L1',
    'SP_D2_343_TB-L2',
    'SP_D2_360_TB-DOC1',
    'SP_D2_360_TB-L1',
    'SP_D2_360_TB-L2',
    'SP_D2_396_TB-DOC1',
    'SP_D2_396_TB-L1',
    'SP_D2_396_TB-L2',
    'SP_DID_018_TB-DOC1',
    'SP_DID_018_TB-L1',
    'SP_DID_137_TB-DOC1',
    'SP_DID_137_TB-L1',
    'SP_DID_161_TB-DOC1',
    'SP_DID_161_TB-L1',
    'SP_DID_208_TB-DOC1',
    'SP_DID_208_TB-L1',
    'SP_DID_235_TB-DOC1',
    'SP_DID_235_TB-L1',
    'SP_DID_242_TB-DOC1',
    'SP_DID_242_TB-L1',
    'SP_DID_250_TB-DOC1',
    'SP_DID_250_TB-L1',
    'SP_DID_251_TB-L1',
    'SP_EF_124_TB-L1',
    'SP_EF_153_TB-L1',
    'SP_EF_156_TB-L1',
    'SP_EF_377_TB-L1',
    'SP_EF_388_TB-L1',
    'SP_EF_405_TB-L1'
}

entoa_prosodic_ds = load_dataset("nilc-nlp/NURC-SP_ENTOA_TTS", name="prosodic", trust_remote_code=True)

# Python wrapper to invoke the given bash script with the given input args
# def run_script(script, args):
#     args = ' \\'.join(args)
#     cmd = f"python {script} \\{args}"

#     print(cmd.replace(" \\", "\n"))
#     print()
#     !$cmd #type: ignore


def filter_dataset_by_speaker(dataset:Dataset, speaker_id:str) -> Dataset:
    """
    Filter a dataset to include only samples from a specific speaker.
    
    Args:
        dataset: A HuggingFace dataset object
        speaker_id: The ID of the speaker to filter for
        
    Returns:
        A filtered dataset containing only the specified speaker's data
    """
    # Create a filter function that checks if the speaker matches the given ID
    def speaker_filter(example):
        return example['speaker'] == speaker_id
    
    # Apply the filter to the dataset
    filtered_dataset = dataset.filter(speaker_filter)
    
    print(f"Original dataset size: {len(dataset)} samples")
    print(f"Filtered dataset size for speaker {speaker_id}: {len(filtered_dataset)} samples")
    
    return filtered_dataset


def convert_wav_sample_rate(input_file:Path, output_file:Path, sample_rate:int) -> bool:
    """
    Convert a WAV file to a different sample rate using pydub.
    
    Args:
        input_file (str): Path to the input WAV file
        output_file (str): Path to save the converted WAV file
        sample_rate (int): Target sample rate in Hz (e.g., 44100, 48000)
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    assert input_file.exists()
    
    try:
        # Load the audio file
        audio:AudioSegment = AudioSegment.from_wav(input_file)

        if audio.frame_rate == sample_rate:
            return True
        
        # Set the new sample rate
        audio = audio.set_frame_rate(sample_rate)
        
        # Export with the new sample rate
        audio.export(output_file, format="wav")
        
        return True
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def create_manifest(dataset:DatasetDict, split:str) -> tuple[Path, dict]:
    manifest_path = AUDIO_ROOT_PATH/f'{split}_manifest.json'
    speakers = dict()
    sid = 0

    if manifest_path.exists():
        os.remove(manifest_path)

    with open(AUDIO_ROOT_PATH/f'{split}_manifest.json', 'w+') as f:
        for sample in dataset:
            audio_path:Path = AUDIO_ROOT_PATH / sample['path']

            assert audio_path.exists()
            assert convert_wav_sample_rate(audio_path, audio_path, 22050)

            speakers[sample['speaker']] = sid

            json.dump({
                "audio_filepath": str(audio_path),
                "duration": float(sample['duration']),
                "text": sample['normalized_text'],
                "speaker": sid
            }, f)

            sid+=1
            f.write('\n')

    return manifest_path, speakers


def main():
    for speaker in COMMON_SPEAKERS:
        print(f"\33[42m Processing speaker: {speaker} \33[0m")
        filtered_train_ds = filter_dataset_by_speaker(entoa_prosodic_ds['train'], speaker)
        manifest_train_path, sp1 = create_manifest(filtered_train_ds, 'train')

        filtered_val_ds = filter_dataset_by_speaker(entoa_prosodic_ds['validation'], speaker)
        manifest_validation_path, sp2 = create_manifest(filtered_val_ds, 'validation')

        # Build the command arguments
        cmd_args = [
            "python", str(FINETUNING_SCRIPT_PATH),
            "--config-name=fastpitch_align_v1.05.yaml",
            f"train_dataset={manifest_train_path}",
            f"validation_datasets={manifest_validation_path}",
            f"sup_data_path={SUP_DATA_PATH}",
            f"phoneme_dict_path={CMU_DICT_PATH}",
            f"heteronyms_path={HETERONYMS_PATH}",
            f"exp_manager.exp_dir={EXP_MANAGER_DIR}",
            f"+init_from_nemo_model={FASTPITCH_PATH}",
            "+trainer.max_steps=1000", "~trainer.max_epochs",
            "trainer.check_val_every_n_epoch=25",
            "model.train_ds.dataloader_params.batch_size=4", "model.validation_ds.dataloader_params.batch_size=24",
            "model.n_speakers=41", "model.pitch_mean=152.3", "model.pitch_std=64.0",
            "model.pitch_fmin=30", "model.pitch_fmax=512", "model.optim.lr=2e-4",
            "~model.optim.sched", "model.optim.name=adam", "trainer.devices=1", "trainer.strategy=auto",
            "+model.text_tokenizer.add_blank_at=true",
        ]

        # Print the command for reference
        print("Executing command:")
        print(" \\\n".join(cmd_args))
        print()

        # Run the command and capture output
        try:
            result = subprocess.run(
                cmd_args,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Command executed successfully")
            print("Output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with return code {e.returncode}")
            print("Error output:")
            print(e.stderr)
        break

if __name__ == "__main__":
    main()