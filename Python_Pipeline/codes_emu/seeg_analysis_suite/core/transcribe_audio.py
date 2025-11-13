# Author: Sunil Mathew
# Date: 13 May 2024
# Transcribe recall audio data using WhisperX, vosk, and Google Cloud Speech-to-Text API
# whisperx and vosk provides word-level time timestamps
# whisperx need hugging face token for being able to use libraries for differentiating speakers

import os
import scipy.io.wavfile
import speech_recognition as sr
try:
    import whisperx
except:
    print('whisperx not installed')

def transcribe_audio(dir_save, audio_data, mic, c):
    fs = 30000 # Sampling frequency in Hz
    audio_file = os.path.join(dir_save, f'{mic}_audio.wav')
    audio_chunk = audio_data # audio_data[:fs*30] # 30 seconds
    scipy.io.wavfile.write(audio_file, fs, audio_chunk)

    # r = sr.Recognizer()
    # with sr.AudioFile(audio_file) as source:
    #     r.adjust_for_ambient_noise(source)
    #     audio = r.record(source)
    try:
        # text = r.recognize_sphinx(audio)
        # text = r.recognize_google(audio)
        text = transcribe_audio_whisperx(dir_save=dir_save, audio_data=audio_data, mic_ch_lbl=mic)
        # print(f'{mic}: {text}')
        # c.log.emit(f'{mic}: {text}')
        # text = transcribe_audio_vosk(dir_save=dir_save, audio_data=audio_data, mic_ch_lbl=mic)
        # text = transcribe_gcs_with_word_time_offsets(gcs_uri=audio_file)
        # print(f'{mic}: {text}')
    except sr.UnknownValueError:
        print(f'{mic}: Could not understand the audio')
    except sr.RequestError as e:
        print(f'{mic}: Could not request results; {e}')

    return text

def transcribe_audio_whisperx(dir_save, audio_data, mic_ch_lbl):
    fs = 30000 # Sampling frequency in Hz
    # remove spaces from mic_ch_lbl
    mic_ch_lbl = mic_ch_lbl.replace(' ', '_')
    audio_file = os.path.join(dir_save, f'{mic_ch_lbl}_audio.wav')
    audio_chunk = audio_data[:fs*30]
    scipy.io.wavfile.write(audio_file, fs, audio_chunk)

    # whisperx
    batch_size = 16
    device = "cpu"
    HF_TOKEN = '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'
    model = whisperx.load_model(whisper_arch="large-v2", 
                                device=device, 
                                compute_type="float32")

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, chunk_size=120, print_progress=True)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs
    return result["segments"]

def transcribe_audio_vosk(dir_save, audio_data, mic_ch_lbl):
    fs = 30000 # Sampling frequency in Hz
    audio_file = os.path.join(dir_save, f'{mic_ch_lbl}_audio.wav')
    audio_chunk = audio_data[:fs*30]
    scipy.io.wavfile.write(audio_file, fs, audio_chunk)

    # vosk
    import wave
    import sys

    from vosk import Model, KaldiRecognizer, SetLogLevel

    # You can set log level to -1 to disable debug messages
    SetLogLevel(0)

    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        sys.exit(1)

    model = Model(lang="en-us")

    # You can also init model by name or with a folder path
    # model = Model(model_name="vosk-model-en-us-0.21")
    # model = Model("models/en")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    while True:
        data = wf.readframes(300000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            print(rec.Result())
            # self.c.log.emit(f'{mic_ch_lbl}: {rec.Result()}')
        else:
            print(rec.PartialResult())
            # self.c.log.emit(f'{mic_ch_lbl}: {rec.PartialResult()}')

    print(rec.FinalResult())

    return rec.FinalResult()

def transcribe_gcs_with_word_time_offsets(gcs_uri: str,):
    """Transcribe the given audio file asynchronously and output the word time
    offsets."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    result = operation.result(timeout=90)

    for result in result.results:
        alternative = result.alternatives[0]
        print(f"Transcript: {alternative.transcript}")
        print(f"Confidence: {alternative.confidence}")

        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time

            print(
                f"Word: {word}, start_time: {start_time.total_seconds()}, end_time: {end_time.total_seconds()}"
            )

    return result