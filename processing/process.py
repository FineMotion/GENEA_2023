from pathlib import Path
from process_audio import *
from process_bvh import *
from tqdm import tqdm
import os

AUDIO_FOLDER = 'data/val/main-agent/wav'
BVH_FOLDER = 'data/val/main-agent/bvh'

RESULT_AUDIO_FOLDER = 'clips_data/val/main-agent/audio'
RESULT_GESTURES_FOLDER = 'clips_data/val/main-agent/gestures'


def process_one_file(file_path, bvh_file_path):
    audio_data = load_audio_file(file_path)
    beats = extract_speech_beats(audio_data)
    # beats = np.unique(beats)
    audio_clips = split_audio_into_clips(audio_data, beats)

    parsed_data = load_bvh_file(bvh_file_path)
    if np.max(beats) > len(parsed_data.values):
        return None, None
        # print(file_path)
        # print(beats)
        # print(len(parsed_data))

    processed_data = data_pipline(parsed_data)
    frame_time = parsed_data.framerate
    bvh_blocks = split_bvh_into_blocks(processed_data, beats, frame_time)

    for i, block in enumerate(bvh_blocks):
        assert block.shape[0] != 0

    """
    print(len(audio_data))
    print(len(parsed_data.values))
    print(beats)
    print(frame_time)
    print((beats / frame_time).astype(int))
    print(len(audio_clips))
    print(len(bvh_blocks))

    print(np.array(audio_clips[0]).shape)
    print(np.array(bvh_blocks[0]).shape)

    print()
    print(np.array(audio_clips[1]).shape)
    print(np.array(bvh_blocks[1]).shape)

    print()
    print(np.array(audio_clips[2]).shape)
    print(np.array(bvh_blocks[2]).shape)
    """

    return np.array(audio_clips, dtype=object), np.array(bvh_blocks, dtype=object)


def main(audio_folder: Path, bvh_folder: Path):
    if not os.path.exists(RESULT_AUDIO_FOLDER):
        os.makedirs(RESULT_AUDIO_FOLDER)
    if not os.path.exists(RESULT_GESTURES_FOLDER):
        os.makedirs(RESULT_GESTURES_FOLDER)
    audio_recordings = list(audio_folder.glob('*.wav')) if audio_folder.is_dir() else [audio_folder]
    bvh_recordings = list(bvh_folder.glob('*.bvh')) if bvh_folder.is_dir() else [bvh_folder]

    audio_recordings = [str(a) for a in audio_recordings]
    bvh_recordings = [str(b) for b in bvh_recordings]

    for audio_record in tqdm(audio_recordings):
        bvh_record = audio_record.replace("wav", "bvh")
        assert bvh_record in bvh_recordings

        audio_clips, bvh_blocks = process_one_file(audio_record, bvh_record)
        # process_one_file(audio_record, bvh_record)

        if audio_clips is None:
            continue

        assert len(audio_clips) == len(bvh_blocks)

        name = audio_record[audio_record.rfind("/") + 1:]

        dst_path = RESULT_AUDIO_FOLDER + "/" + name.replace('.wav', '.npy')
        np.save(str(dst_path), audio_clips)
        dst_path = RESULT_GESTURES_FOLDER + "/" + name.replace('.wav', '.npy')
        np.save(str(dst_path), bvh_blocks)

    return


if __name__ == '__main__':
    file_path = 'data/trn/interloctr/wav/trn_2023_v0_000_interloctr.wav'
    bvh_file_path = 'data/trn/interloctr/bvh/trn_2023_v0_000_interloctr.bvh'
    main(Path(AUDIO_FOLDER), Path(BVH_FOLDER))
