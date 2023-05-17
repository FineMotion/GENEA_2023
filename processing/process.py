from pathlib import Path
from process_audio import *
from process_bvh import *
from process_tsv import *
from tqdm import tqdm
import os

MODE = "trn"

AUDIO_FOLDER = os.path.join("data", MODE, "main-agent/wav")
BVH_FOLDER = os.path.join("data", MODE, "main-agent/bvh")
TSV_FOLDER = os.path.join("data", MODE, "main-agent/tsv")

SPLIT_AUDIO_FOLDER = os.path.join("split_data", MODE, "main-agent/audio")
SPLIT_GESTURES_FOLDER = os.path.join("split_data", MODE, "main-agent/gestures")

RESULT_AUDIO_FOLDER = os.path.join("clips_data", MODE, "main-agent/audio")
RESULT_GESTURES_FOLDER = os.path.join("clips_data", MODE, "main-agent/gestures")


def process_one_file(audio_path, gestures_path):
    audio_data = np.load(audio_path)
    processed_data = np.load(gestures_path)

    beats = extract_speech_beats(audio_data)
    audio_clips = split_audio_into_clips(audio_data, beats)
    bvh_blocks = split_bvh_into_blocks(processed_data, beats)

    for i, block in enumerate(bvh_blocks):
        assert block.shape[0] != 0

    return np.array(audio_clips, dtype=object), np.array(bvh_blocks, dtype=object)


def save_data(data, file_path):
    dir = file_path[:file_path.rfind("/")]
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(str(file_path), data)


def split_one_speaker(file_path, bvh_file_path, tsv_file_path):
    audio_data = load_audio_file(file_path)
    tsv_data = load_tsv_file(tsv_file_path)
    borders = get_borders(tsv_data, pause_threshold=0.5) * SAMPLE_RATE
    borders_of_silence = get_borders_of_silence(audio_data)
    borders = erase_borders_of_silence(borders, borders_of_silence)
    parsed_data = load_bvh_file(bvh_file_path)
    processed_data = data_pipline(parsed_data)

    if len(audio_data) / HOP_LENGTH != processed_data.shape[0]:
        return

    name = file_path[file_path.rfind("/") + 1:]

    for i in range(len(borders)):
        start = borders[i][0]
        end = borders[i][1]

        cur_audio = audio_data[start: end]
        start //= HOP_LENGTH
        end //= HOP_LENGTH
        cur_gestures = processed_data[start: end]

        dst_path = SPLIT_AUDIO_FOLDER + "/" + name.replace('.wav', "/" + str(i) + ".npy")
        save_data(cur_audio, str(dst_path))

        dst_path = SPLIT_GESTURES_FOLDER + "/" + name.replace('.wav', "/" + str(i) + ".npy")
        save_data(cur_gestures, str(dst_path))


def main(audio_folder: Path, bvh_folder: Path, tsv_folder: Path):
    if not os.path.exists(RESULT_AUDIO_FOLDER):
        os.makedirs(RESULT_AUDIO_FOLDER)
    if not os.path.exists(RESULT_GESTURES_FOLDER):
        os.makedirs(RESULT_GESTURES_FOLDER)

    audio_recordings = list(audio_folder.glob('*.wav')) if audio_folder.is_dir() else [audio_folder]
    bvh_recordings = list(bvh_folder.glob('*.bvh')) if bvh_folder.is_dir() else [bvh_folder]
    tsv_recordings = list(tsv_folder.glob('*.tsv')) if tsv_folder.is_dir() else [tsv_folder]

    audio_recordings = [str(a) for a in audio_recordings]
    bvh_recordings = [str(b) for b in bvh_recordings]
    tsv_recordings = [str(b) for b in tsv_recordings]

    for audio_record in tqdm(audio_recordings):
        print(audio_record)
        bvh_record = audio_record.replace("wav", "bvh")
        tsv_record = audio_record.replace("wav", "tsv")
        assert bvh_record in bvh_recordings
        assert tsv_record in tsv_recordings

        split_one_speaker(audio_record, bvh_record, tsv_record)
        name = audio_record[audio_record.rfind("/") + 1:audio_record.rfind(".")]
        split_audio_folder = Path(os.path.join(SPLIT_AUDIO_FOLDER, name))
        split_bvh_folder = Path(os.path.join(SPLIT_GESTURES_FOLDER, name))

        audio_clips = list(split_audio_folder.glob('*.npy'))
        bvh_clips = list(split_bvh_folder.glob('*.npy'))

        audio_split_recordings = [str(a) for a in audio_clips]
        bvh_split_recordings = [str(b) for b in bvh_clips]

        for audio in audio_split_recordings:
            gestures = audio.replace("audio", "gestures")
            assert gestures in bvh_split_recordings
            audio_clips, bvh_blocks = process_one_file(audio, gestures)
            if audio_clips is None:
                continue

            assert len(audio_clips) == len(bvh_blocks)

            dst_path = audio.replace('split_data', 'clip_data')
            save_data(audio_clips, dst_path)
            dst_path = gestures.replace('split_data', 'clip_data')
            save_data(bvh_blocks, dst_path)


if __name__ == '__main__':
    main(Path(AUDIO_FOLDER), Path(BVH_FOLDER), Path(TSV_FOLDER))
