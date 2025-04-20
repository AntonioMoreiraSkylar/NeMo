from datasets import load_dataset

cml_ds = load_dataset("ylacombe/cml-tts", "portuguese")
entoa_prosodic_ds = load_dataset("nilc-nlp/NURC-SP_ENTOA_TTS", name="prosodic", trust_remote_code=True)