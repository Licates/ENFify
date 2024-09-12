"""Module with util funcions for data augmentation like making an authentic
audio tampered or chunk larger audios."""

import os
import re
import sys
from glob import glob

import numpy as np
from loguru import logger
from scipy.io import wavfile
from tqdm import tqdm

from enfify.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR
from enfify.utils import zero_pad_number_in_filename

logger.remove()
logger.add(sys.stdout, level="INFO")
np.random.seed(42)


def augmentation(
    dataset_dir, interim_dir, clip_length, max_cutlen, num_clips=None, regex_pattern: str = None
):
    if os.path.exists(interim_dir):
        logger.warning(f"Directory {interim_dir} already exists, skipping")
        return

    if not os.path.exists(dataset_dir):
        logger.error(f"Directory {dataset_dir} does not exist")
        return

    files = sorted(glob(str(dataset_dir / "**" / "*.wav"), recursive=True))
    if regex_pattern:
        regex_pattern = re.compile(regex_pattern)
        files = [f for f in files if regex_pattern.match(os.path.basename(f))]
    logger.info(f"Found {len(files)} files for {os.path.basename(interim_dir)}")

    process_files(
        files=files,
        interim_dir=interim_dir,
        clip_length=clip_length,
        num_clips=num_clips,
        max_cutlen=max_cutlen,
    )


def process_files(files, interim_dir, clip_length, num_clips, max_cutlen):
    os.makedirs(interim_dir, exist_ok=True)

    calc_clips = num_clips is None

    for file in tqdm(files):
        basename = zero_pad_number_in_filename(os.path.splitext(os.path.basename(file))[0], 2)

        logger.debug(f"Reading file {file}")
        rate, data = wavfile.read(file)
        logger.debug(f"Rate: {rate}, data shape: {data.shape}")

        cliplen_samples = round(clip_length * rate)
        max_cutlen_samples = round(max_cutlen * rate)

        if calc_clips:
            num_clips = np.ceil(len(data) / cliplen_samples).astype(int)

        if len(data) > cliplen_samples * num_clips + max_cutlen_samples:
            logger.warning(
                f"Data of {os.path.basename(file)} is longer ({round(len(data)*rate)}) than all clips combined ({clip_length*num_clips}), not all data will be used"
            )

        # Convert stereo to mono by averaging channels
        if data.ndim == 2:
            data = np.mean(data, axis=1)

        logger.debug(f"Segmenting {basename} into {num_clips} clips")
        clips = segment_in_clips(data, cliplen_samples + max_cutlen_samples, num_clips)
        clips = [clip for clip in clips if clip is not None]

        logger.debug(f"Saving {basename} clips")
        for i, clip in enumerate(clips):
            # Save authentic data
            auth_path = interim_dir / f"{basename}-{i:02}-auth.wav"
            authentic_clip = clip[:cliplen_samples]
            wavfile.write(auth_path, rate, authentic_clip)

            # Save tampered data
            tamp_path = interim_dir / f"{basename}-{i:02}-tamp.wav"
            tampered_clip = random_cut_clip(clip, max_cutlen_samples)
            tampered_clip = tampered_clip[:cliplen_samples]
            wavfile.write(tamp_path, rate, tampered_clip)


def segment_in_clips(data, cliplen_samples, num_clips):
    """Segment a data array into a number of clips of a given length."""

    if len(data) < cliplen_samples:
        logger.error(f"Data is shorter than clip")
        return

    clip_starts = np.linspace(0, len(data) - cliplen_samples, num_clips).astype(int)
    clips = [data[start : start + cliplen_samples] for start in clip_starts]
    return clips


def random_cut_clip(clip, max_cutlen_samples):
    """Cut a random segment from a clip."""
    cutlen = np.random.randint(max_cutlen_samples) + 1
    start = np.random.randint(0, len(clip) - cutlen)
    cutted_clip = np.delete(clip, slice(start, start + cutlen))
    return cutted_clip


if __name__ == "__main__":
    augmentation(
        EXTERNAL_DATA_DIR / "Carioca" / "BASE CARIOCA 1",
        INTERIM_DATA_DIR / "Carioca1",
        clip_length=10,
        num_clips=10,
        max_cutlen=2,
        regex_pattern=r"^(HC|MC)\d+\.wav$",
    )
    augmentation(
        EXTERNAL_DATA_DIR / "ENF-WHU-Dataset" / "H1",
        INTERIM_DATA_DIR / "WHU",
        clip_length=10,
        num_clips=None,
        max_cutlen=2,
    )
    augmentation(
        EXTERNAL_DATA_DIR / "ENF-WHU-Dataset" / "H1_ref",
        INTERIM_DATA_DIR / "WHU_ref",
        clip_length=10,
        num_clips=None,
        max_cutlen=2,
    )

    augmentation(
        EXTERNAL_DATA_DIR / "Carioca" / "BASE CARIOCA 1",
        INTERIM_DATA_DIR / "Carioca1_min",
        clip_length=60,
        num_clips=None,
        max_cutlen=2,
        regex_pattern=r"^(HC|MC)\d+\.wav$",
    )
    augmentation(
        EXTERNAL_DATA_DIR / "ENF-WHU-Dataset" / "H1",
        INTERIM_DATA_DIR / "WHU_min",
        clip_length=60,
        num_clips=None,
        max_cutlen=2,
    )
    augmentation(
        EXTERNAL_DATA_DIR / "ENF-WHU-Dataset" / "H1_ref",
        INTERIM_DATA_DIR / "WHU_ref_min",
        clip_length=60,
        num_clips=None,
        max_cutlen=2,
    )
