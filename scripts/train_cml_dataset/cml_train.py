import os
import json
import torch
import logging
import subprocess
from pathlib import Path
from typing import List

import numpy as np  
from scipy.io import wavfile
from datasets import load_dataset, DatasetDict

from nemo.collections.tts.models import HifiGanModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

Logger = logging.getLogger()

os.environ["HYDRA_FULL_ERROR"] = "1"

def run_script(script, args:List[str]):
    cmd = ["python", script] + args
    subprocess.run(cmd)


NEMO_ROOT_DIR = Path(os.getenv("NEMO_ROOT_DIR"))
#DATA_ROOT = Path("/mnt/fs/amoreira/data/cml-portuguese/exps/HifiGan/test_run/")
DATA_ROOT = Path(os.getenv("DATA_ROOT"))
DATA_DIR = DATA_ROOT/'cml-portuguese'
AUDIO_DIR = DATA_DIR/'audio'

NEMO_DIR = Path(NEMO_ROOT_DIR)
NEMO_EXAMPLES_DIR = NEMO_DIR / "examples" / "tts"
NEMO_CONFIG_DIR = NEMO_EXAMPLES_DIR / "conf"
NEMO_SCRIPT_DIR = NEMO_DIR / "scripts" / "dataset_processing" / "tts"

PHONEMES_ENTOA = Path(__file__).parent.resolve() / 'entoa_tts_pros_333.txt'
HETERONYMS_ENTOA = Path(__file__).parent.resolve() / 'heteronyms_entoa.txt'

print(
    NEMO_DIR, 
    NEMO_EXAMPLES_DIR,
    NEMO_CONFIG_DIR,
    NEMO_SCRIPT_DIR,
    sep='\n'
)

assert all((
    NEMO_DIR.exists(), NEMO_EXAMPLES_DIR.exists(), NEMO_CONFIG_DIR.exists(),
    NEMO_SCRIPT_DIR.exists(), DATA_ROOT.exists(), PHONEMES_ENTOA.exists(), HETERONYMS_ENTOA.exists(),
    )), 'Required paths does not exists.'


def ndarray_to_wav(array:np.ndarray, filename:str, filepath:Path, sample_rate:int) -> None:
    wavfile.write(
        filename=filepath/filename,
        rate=sample_rate,
        data=array
    )


def ensure_folders():
    if not DATA_ROOT.exists():
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not AUDIO_DIR.exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def create_manifest(dataset:DatasetDict, split:str):
    with open(DATA_DIR/f'{split}.json', 'w') as f:
        for sample in dataset[split].select(range(20)):
            audio = sample['audio'] # type: ignore

            ndarray_to_wav(
                array=audio['array'],
                filename=audio['path'],
                filepath=AUDIO_DIR,
                sample_rate=audio['sampling_rate']
            )

            # {"audio_filepath": str, "duration": float, "text": str, "speaker": 225}

            try:
                sample_text = sample['text'].lower()
            except AttributeError:
                print(sample)
                continue

            json.dump({
                "audio_filepath": audio['path'],
                "duration": sample['duration'], # type: ignore
                "text": sample['text'].lower(), # type: ignore
                "speaker": sample['speaker_id'] # type: ignore
            },f, ensure_ascii=False)
            
            f.write('\n')


def update_metadata(data_type):
    input_filepath = DATA_DIR / f"{data_type}.json"
    output_filepath = DATA_DIR / f"{data_type}_raw.json"

    entries = read_manifest(input_filepath)
    for entry in entries:
        # Provide relative path instead of absolute path
        entry["audio_filepath"] = entry["audio_filepath"].replace("audio/", "")
        # Prepend speaker ID with the name of the dataset: 'cml'
        entry["speaker"] = f"cml_{entry['speaker']}"

    write_manifest(output_path=output_filepath, target_manifest=entries, ensure_ascii=False)


def audio_processing(data_type:str):
    audio_preprocessing_script = NEMO_SCRIPT_DIR / "preprocess_audio.py"
    # Directory with raw audio data
    input_audio_dir = DATA_DIR / "audio"
    # Directory to write preprocessed audio to
    output_audio_dir = DATA_DIR / "audio_preprocessed"
    # Whether to overwrite existing audio, if it exists in the output directory
    overwrite_audio = True
    # Whether to overwrite output manifest, if it exists
    overwrite_manifest = True
    # Number of threads to parallelize audio processing across
    num_workers = 32
    # Downsample data from 48khz to 44.1khz for compatibility
    output_sample_rate = 22050
    # Format of output audio files. Use "flac" to compress to a smaller file size.
    output_format = "flac"
    # Method for silence trimming. Can use "energy.yaml" or "vad.yaml".
    # We use VAD for VCTK because the audio has background noise.
    trim_config_path = NEMO_CONFIG_DIR / "trim" / "vad.yaml"
    # Volume level (0, 1] to normalize audio to
    volume_level = 0.95
    # Filter out audio shorter than min_duration or longer than max_duration seconds.
    # We set these bounds relatively low/high, as we can place stricter limits at training time
    min_duration = 0.25
    max_duration = 30.0
    # Output file with entries that are filtered out based on duration
    filter_file = DATA_DIR / "filtered.json"

    input_filepath = DATA_DIR / f"{data_type}_raw.json"
    output_filepath = DATA_DIR / f"{data_type}_manifest.json"

    args = [
        f"--input_manifest={input_filepath}",
        f"--output_manifest={output_filepath}",
        f"--input_audio_dir={input_audio_dir}",
        f"--output_audio_dir={output_audio_dir}",
        f"--num_workers={num_workers}",
        f"--output_sample_rate={output_sample_rate}",
        f"--output_format={output_format}",
        f"--trim_config_path={trim_config_path}",
        f"--volume_level={volume_level}",
        f"--min_duration={min_duration}",
        f"--max_duration={max_duration}",
        f"--filter_file={filter_file}",
    ]

    if overwrite_manifest:
        args.append("--overwrite_manifest")
    if overwrite_audio:
        args.append("--overwrite_audio")

    run_script(audio_preprocessing_script, args)


def speaker_mapping():
    speaker_map_script = NEMO_SCRIPT_DIR / "create_speaker_map.py"

    train_manifest_filepath = DATA_DIR / "train_manifest.json"
    dev_manifest_filepath = DATA_DIR / "dev_manifest.json"
    speaker_filepath = DATA_DIR / "speakers.json"

    args = [
        f"--manifest_path={train_manifest_filepath}",
        f"--manifest_path={dev_manifest_filepath}",
        f"--speaker_map_path={speaker_filepath}"
    ]

    run_script(speaker_map_script, args)


def feature_computation(data_type:str):
    feature_script = NEMO_SCRIPT_DIR / "compute_features.py"

    sample_rate = 22050

    if sample_rate == 22050:
        feature_config_filename = "feature_22050.yaml"
    elif sample_rate == 44100:
        feature_config_filename = "feature_44100.yaml"
    else:
        raise ValueError(f"Unsupported sampling rate {sample_rate}")

    feature_config_path = NEMO_CONFIG_DIR / "feature" / feature_config_filename
    audio_dir = DATA_DIR / "audio_preprocessed"
    feature_dir = DATA_DIR / "features"
    num_workers = 32

    input_filepath = DATA_DIR / f"{data_type}_manifest.json"

    args = [
        f"--feature_config_path={feature_config_path}",
        f"--manifest_path={input_filepath}",
        f"--audio_dir={audio_dir}",
        f"--feature_dir={feature_dir}",
        f"--num_workers={num_workers}",
    ]

    run_script(feature_script, args)


def feature_statistics():
    feature_stats_script = NEMO_SCRIPT_DIR / "compute_feature_stats.py"

    train_manifest_filepath = DATA_DIR / "train_manifest.json"
    dev_manifest_filepath = DATA_DIR / "dev_manifest.json"
    output_stats_path = DATA_DIR / "feature_stats.json"

    sample_rate = 22050

    if sample_rate == 22050:
        feature_config_filename = "feature_22050.yaml"
    elif sample_rate == 44100:
        feature_config_filename = "feature_44100.yaml"
    else:
        raise ValueError(f"Unsupported sampling rate {sample_rate}")

    feature_config_path = NEMO_CONFIG_DIR / "feature" / feature_config_filename
    audio_dir = DATA_DIR / "audio_preprocessed"
    feature_dir = DATA_DIR / "features"

    args = [
        f"--feature_config_path={feature_config_path}",
        f"--manifest_path={train_manifest_filepath}",
        f"--manifest_path={dev_manifest_filepath}",
        f"--audio_dir={audio_dir}",
        f"--audio_dir={audio_dir}",
        f"--feature_dir={feature_dir}",
        f"--feature_dir={feature_dir}",
        f"--stats_path={output_stats_path}",
    ]

    run_script(feature_stats_script, args)

def hifi_gan_training():
    dataset_name = "cml"
    audio_dir = DATA_DIR / "audio_preprocessed"
    train_manifest_filepath = DATA_DIR / "train_manifest.json"
    dev_manifest_filepath = DATA_DIR / "dev_manifest.json"

    hifigan_training_script = NEMO_EXAMPLES_DIR / "hifigan.py"

    # The total number of training steps will be (epochs * steps_per_epoch)
    epochs = 10
    steps_per_epoch = 10

    sample_rate = 22050

    # Config files specifying all HiFi-GAN parameters
    hifigan_config_dir = NEMO_CONFIG_DIR / "hifigan_dataset"

    if sample_rate == 22050:
        hifigan_config_filename = "hifigan_22050.yaml"
    elif sample_rate == 44100:
        hifigan_config_filename = "hifigan_44100.yaml"
    else:
        raise ValueError(f"Unsupported sampling rate {sample_rate}")

    # Name of the experiment that will determine where it is saved locally and in TensorBoard and WandB
    run_id = "test_run"
    exp_dir = DATA_DIR / "exps"
    hifigan_exp_output_dir = exp_dir / "HifiGan" / run_id
    # Directory where predicted audio will be stored periodically throughout training
    hifigan_log_dir = hifigan_exp_output_dir / "logs"

    if torch.cuda.is_available():
        accelerator="gpu"
        batch_size = 16
    else:
        accelerator="cpu"
        batch_size = 2

    args = [
        f"--config-path={hifigan_config_dir}",
        f"--config-name={hifigan_config_filename}",
        f"max_epochs={epochs}",
        f"weighted_sampling_steps_per_epoch={steps_per_epoch}",
        f"batch_size={batch_size}",
        f"log_dir={hifigan_log_dir}",
        f"exp_manager.exp_dir={exp_dir}",
        f"+exp_manager.version={run_id}",
        f"trainer.accelerator={accelerator}",
        f"+train_ds_meta.{dataset_name}.manifest_path={train_manifest_filepath}",
        f"+train_ds_meta.{dataset_name}.audio_dir={audio_dir}",
        f"+val_ds_meta.{dataset_name}.manifest_path={dev_manifest_filepath}",
        f"+val_ds_meta.{dataset_name}.audio_dir={audio_dir}",
        f"+log_ds_meta.{dataset_name}.manifest_path={dev_manifest_filepath}",
        f"+log_ds_meta.{dataset_name}.audio_dir={audio_dir}"
    ]

    run_script(hifigan_training_script, args)

    return hifigan_exp_output_dir


def fast_pitch_training(run_id: str, hifigan_exp_output_dir:Path):
    dataset_name = "cml"
    audio_dir = DATA_DIR / "audio_preprocessed"
    train_manifest_filepath = DATA_DIR / "train_manifest.json"
    dev_manifest_filepath = DATA_DIR / "dev_manifest.json"
    fastpitch_training_script = NEMO_EXAMPLES_DIR / "fastpitch.py"

    # The total number of training steps will be (epochs * steps_per_epoch)
    epochs = 10
    steps_per_epoch = 40_000

    with open(DATA_DIR/'speakers.json', 'r') as f:
        speakers = json.load(f)
        assert isinstance(speakers, dict)
        print(f">>>>> Number of Speakers: {len(speakers)}")

    num_speakers = len(speakers)
    sample_rate = 22050

    # Config files specifying all FastPitch parameters
    # NeMo/examples/tts/conf/fastpitch
    fastpitch_config_dir = NEMO_CONFIG_DIR / "fastpitch"

    if sample_rate == 22050:
        fastpitch_config_filename = "fastpitch_22050.yaml"
    elif sample_rate == 44100:
        fastpitch_config_filename = "fastpitch_44100.yaml"
    else:
        raise ValueError(f"Unsupported sampling rate {sample_rate}")

    # Metadata files and directories
    dataset_file_dir = NEMO_DIR / "scripts" / "tts_dataset_files"

    speaker_path = DATA_DIR / "speakers.json"
    feature_dir = DATA_DIR / "features"
    stats_path = DATA_DIR / "feature_stats.json"

    def get_latest_checkpoint(checkpoint_dir):
        output_path = None
        for checkpoint_path in checkpoint_dir.iterdir():
            checkpoint_name = str(checkpoint_path.name)
            if checkpoint_name.endswith(".nemo"):
                output_path = checkpoint_path
                break
            if checkpoint_name.endswith("last.ckpt"):
                output_path = checkpoint_path

        if not output_path:
            raise ValueError(f"Could not find latest checkpoint in {checkpoint_dir}")

        return output_path

    # HiFi-GAN model for generating audio predictions from FastPitch output
    vocoder_type = "hifigan"
    vocoder_checkpoint_path = get_latest_checkpoint(hifigan_exp_output_dir / "checkpoints")

    exp_dir = DATA_DIR / "exps"
    fastpitch_exp_output_dir = exp_dir / "FastPitch" / run_id
    fastpitch_log_dir = fastpitch_exp_output_dir / "logs"

    if torch.cuda.is_available():
        accelerator="gpu"
        batch_size = 32
    else:
        accelerator="cpu"
        batch_size = 4

    args = [
        f"--config-path={fastpitch_config_dir}",
        f"--config-name={fastpitch_config_filename}",
        f"n_speakers={num_speakers}",
        f"speaker_path={speaker_path}",
        f"max_epochs={epochs}",
        f"weighted_sampling_steps_per_epoch={steps_per_epoch}",
        f"phoneme_dict_path={PHONEMES_ENTOA}",
        f"heteronyms_path={HETERONYMS_ENTOA}",
        f"feature_stats_path={stats_path}",
        f"log_dir={fastpitch_log_dir}",
        f"vocoder_type={vocoder_type}",
        f"vocoder_checkpoint_path='{vocoder_checkpoint_path}'",
        f"trainer.accelerator={accelerator}",
        f"exp_manager.exp_dir={exp_dir}",
        f"+exp_manager.version={run_id}",
        f"+train_ds_meta.{dataset_name}.manifest_path={train_manifest_filepath}",
        f"+train_ds_meta.{dataset_name}.audio_dir={audio_dir}",
        f"+train_ds_meta.{dataset_name}.feature_dir={feature_dir}",
        f"+val_ds_meta.{dataset_name}.manifest_path={dev_manifest_filepath}",
        f"+val_ds_meta.{dataset_name}.audio_dir={audio_dir}",
        f"+val_ds_meta.{dataset_name}.feature_dir={feature_dir}",
        f"+log_ds_meta.{dataset_name}.manifest_path={dev_manifest_filepath}",
        f"+log_ds_meta.{dataset_name}.audio_dir={audio_dir}",
        f"+log_ds_meta.{dataset_name}.feature_dir={feature_dir}"
    ]

    run_script(fastpitch_training_script, args)

def pipeline():
    cml_ds = load_dataset("ylacombe/cml-tts", "portuguese")
    ensure_folders()

    #Create Manifest
    create_manifest(cml_ds, 'train') # type: ignore
    create_manifest(cml_ds, 'dev') # type: ignore

    #Update Manifest
    update_metadata("dev")
    update_metadata("train")

    #Audio Processing
    audio_processing("dev")
    audio_processing("train")

    #Speaker Mapping
    speaker_mapping()

    #Feature Computation
    feature_computation("dev")
    feature_computation("train")

    #Feature Statistcs
    feature_statistics()

    #Hifi Traning
    hifigan_exp_output_dir = Path("/home/antonio/models/hifigan/")

    #FastPitch Training
    fast_pitch_training("fast_run", hifigan_exp_output_dir)
    
if __name__ == '__main__':
    pipeline()