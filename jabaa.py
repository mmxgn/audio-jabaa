import glob, os
import librosa
import numpy as np

def constant_power_fade_envelopes(duration:float,
                                  sampleRate:int,
                                  t0:float,
                                  t1:float,) -> np.ndarray:
    """
        Returns two envelopes of duration `duration'
        where fading two signals using them will result
        in a constant power signal.

        Arguments:
            duration (float): The duration of the envelopes in seconds
            sampleRate (int): The samplerate of the signals (must be
                              the same for both signals)
            t0       (float): Time in seconds the first crossfade happens
            t1       (float): Time in seconds before the end the crossfade
                              happens.

        Returns:
            env1, env2 (np.ndarray): The two envelopes
    """
    y = np.ones(int(duration*sampleRate))
    y[:int(t0*sampleRate)] = 1/t0*np.linspace(0, t0, int(t0*sampleRate))
    y[-int(t1*sampleRate):] = 1 - 1/t1*np.linspace(0, t1, int(t1*sampleRate))

    even = 0.5*(np.sqrt(0.5+0.5*y) + np.sqrt(0.5-0.5*y))
    odd = 0.5*(np.sqrt(0.5+0.5*y) +- np.sqrt(0.5-0.5*y))

    return even + odd, even-odd


def file_generator(files: list,
                   segment_duration: float,
                   sampleRate: int,
                   db_thr: float = 30,
                   frame_length: int = 512,
                   hop_length: int = 128,
                   ) -> None:
    """
        Segment generator for audio files from a list of files. Yields segments of
        `segment_duration` each time.

        Arguments:

        files             (list): list of `str'. A list of files to generate data from.
        segment_duration (float): segment duration (in seconds) for each yielded segment.
        sampleRate         (int): sample rate of loaded segments.
        db_thr           (float): threshold below which to skip segments form audio.
        frame_length       (int): frame length for silence analyzer.
        hop_length         (int): hop length for silence analyzer.
    """

    I = 0
    J = 0

    segment = np.zeros((int(segment_duration*sampleRate),))

    k = 0
    file_no = 0

    while True:
        if I >= len(segment):
            yield segment
            segment = np.zeros((int(segment_duration*sampleRate),))
            I = 0

        if k == 0 or J >= len(y):
            J = 0
            y, sr = librosa.core.load(files[file_no], mono=True, sr=sampleRate)
            file_no += 1

            if file_no == len(files):
                break

            # Normalize
            y = y/y.max()

            # Figure out intervals of non-silence (NOTE: Is the threshold right? -- 60db quiet)
            intervals = librosa.effects.split(y, frame_length=frame_length, hop_length=hop_length, top_db=db_thr)

            # Remix according to those intervals
            y = librosa.effects.remix(y, intervals)

        if len(segment[I:]) >= len(y[J:]):
            segment[I:I+len(y[J:])] = y[J:]
            I = I + len(y[J:])
            J = J + len(y[J:])
        else:
            segment[I:] = y[J:J+len(segment[I:])]
            J = J + len(segment[I:])
            I = I + len(segment[I:])
        k += 1

def augment_segment(audio: np.ndarray,
                    sampleRate: int,
                    gain: float or list or tuple or None = None,
                    time_stretch: float or list or tuple or None = None,
                    time_shift: float or list or tuple or None = None,
                    pitch_shift: float or list or tuple or None = None,
                    snr_db: float or list or tuple or None = None,
                   ) -> np.ndarray:
    # Store original lenghth
    orig_len = len(audio)

    # 1. Compute Gain
    if gain is None:
        gain = 0.0
    elif type(gain) in [float, int]:
        gain = gain
    elif type(gain) in [tuple, list]:
        gain = np.random.uniform(gain[0], gain[1])

    # 2. Compute Stretch
    if time_stretch is None:
        time_stretch = 1.0
    elif type(time_stretch) in [float, int]:
        time_stretch = time_stretch
    elif type(time_stretch) in [tuple, list]:
        time_stretch = np.random.uniform(time_stretch[0], time_stretch[1])

    # 3. Compute  Time Shift
    if time_shift is None:
        time_shift = 0.0
    elif type(time_shift) in [float, int]:
        time_shift = time_shift
    elif type(time_shift) in [tuple, list]:
        time_shift = np.random.uniform(time_shift[0], time_shift[1])

    # 4. ComputePitch Shift
    if pitch_shift is None:
        pitch_shift = 0.0
    elif type(pitch_shift) in [float, int]:
        pitch_shift = pitch_shift
    elif type(pitch_shift) in [tuple, list]:
        pitch_shift = np.random.randint(pitch_shift[0], pitch_shift[1])

    # 5. Compute Noise
    if snr_db is None:
        snr_db = None
    elif type(snr_db) in [float, int]:
        snr_db = snr_db
    elif type(snr_db) in [tuple, list]:
        snr_db = np.random.uniform(snr_db[0], snr_db[1])

    # 6. Apply gain
    audio *= 10**(gain/10)

    #7. Apply time stretch
    if time_stretch != 1.0:
        audio = librosa.effects.time_stretch(audio, time_stretch)

    # 8. Apply Time shift
    if time_shift == 0.0 or time_shift is None:
        pass
    elif time_shift > 0.0:
        clean_audio = np.zeros_like(audio)
        clean_audio[int(time_shift*sampleRate):] = audio[:len(audio)-int(time_shift*sampleRate)]
        audio = clean_audio
    elif time_shift < 0.0:
        clean_audio = np.zeros_like(audio)
        shifted_audio = audio[int(time_shift*sampleRate):]
        clean_audio[:len(shifted_audio)] = shifted_audio
        audio = clean_audio

    # 9. Apply Pitch Shift
    if pitch_shift != 0.0:
        audio = librosa.effects.pitch_shift(audio, sampleRate, pitch_shift)

    if len(audio) > orig_len:
        audio = audio[:orig_len]
    elif len(audio) < orig_len:
        original = audio
        audio = np.zeros((orig_len,))
        audio[:len(original)] = original

    # 10. Apply AWGN
    # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    if snr_db is None:
        awgn = np.zeros_like(audio)
    else:
        # Compute signal db
        sig_avg_db = 10*np.log10(np.mean(audio**2))
        noise_std = np.sqrt(10 ** ((sig_avg_db - snr_db) / 10))
        awgn = np.random.randn(*audio.shape)*noise_std

    # Add noise
    audio += awgn
    return audio

def noise_with_db(len_segment: int, db: float) -> np.ndarray:
    noise_std = np.sqrt(10**(db/10))
    return np.random.randn(len_segment)*noise_std

def power(sig: np.ndarray) -> float:
    return 10*np.log10(np.mean(sig**2))

def snr_db(sig: np.ndarray, bg: np.ndarray) -> float:
    return 10*np.log10(np.mean(sig**2)/np.mean(bg**2))

def compute_bg_gain_for_snr(snr: float, sig: np.ndarray, bg: np.ndarray) -> float:
    return np.sqrt(10**((power(sig) - snr)/10)/np.abs(bg**2).mean())
