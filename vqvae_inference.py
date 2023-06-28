from argparse import ArgumentParser

import torch

from vqvae_utils.utils import *
import warnings
from tqdm import tqdm
from src.vqvae.system import VQVAESystem

MAX_BEATS_LEN = 18

warnings.filterwarnings('ignore')


def get_all_data(audio_main_path):
    assert "main-agent" in audio_main_path
    audio_interloctr_path = audio_main_path.replace("main-agent", "interloctr")

    tsv_main_path = audio_main_path.replace("wav", "tsv")
    tsv_interloctr_path = tsv_main_path.replace("main-agent", "interloctr")

    bvh_interloctr_path = audio_main_path.replace("wav", "bvh").replace("main-agent", "interloctr")

    audio_main = load_audio_file(audio_main_path)
    audio_interloctr = load_audio_file(audio_interloctr_path)

    tsv_main = load_tsv_file(tsv_main_path)
    tsv_interloctr = load_tsv_file(tsv_interloctr_path)

    pipeline = create_data_pipeline()
    bvh_interloctr = process_data_pipeline(load_bvh_file(bvh_interloctr_path), pipeline)

    return audio_main, audio_interloctr, tsv_main, tsv_interloctr, bvh_interloctr


def get_invert_borders(borders, audio_len):
    invert_borders = []
    if borders[0][0] != 0:
        invert_borders.append([0, borders[0][0]])

    for i in range(len(borders) - 1):
        invert_borders.append([borders[i][1], borders[i + 1][0]])

    if borders[-1][1] != audio_len:
        invert_borders.append([borders[-1][1], audio_len])

    return np.array(invert_borders)


def get_split_by_order(split_audio_main, split_audio_interloctr, split_gestures_main, split_gestures_interloctr,
                       is_main_first):
    split_audio_first = split_audio_interloctr
    split_audio_second = split_audio_main

    split_gestures_first = split_gestures_interloctr
    split_gestures_second = split_gestures_main

    if is_main_first:
        split_audio_first = split_audio_main
        split_audio_second = split_audio_interloctr

        split_gestures_first = split_gestures_main
        split_gestures_second = split_gestures_interloctr

    return split_audio_first, split_audio_second, split_gestures_first, split_gestures_second


def pad_gestures(gestures, max_frames=18):
    vectors_padded = []
    for g in gestures:
        g_t = torch.from_numpy(g.T)
        ones = torch.ones(g_t.shape[0], max_frames - g_t.shape[1])
        last_val = g_t[:, -1].unsqueeze(1)
        last_val = last_val.expand_as(ones)

        vectors_padded.append(torch.cat([g_t, ones * last_val], dim=1))
    return torch.stack(vectors_padded)


def get_gestures(audio_main_path):
    audio_main, audio_interloctr, tsv_main, tsv_interloctr, bvh_interloctr = get_all_data(audio_main_path)
    assert len(audio_main) == len(audio_interloctr)

    audio_main = audio_main[: len(bvh_interloctr) * HOP_LENGTH]
    audio_interloctr = audio_interloctr[: len(bvh_interloctr) * HOP_LENGTH]

    borders_main = (get_borders(tsv_main, pause_threshold=0.5) * SAMPLE_RATE).astype(np.int)
    borders_main = [b for b in borders_main if b[1] <= len(bvh_interloctr) * HOP_LENGTH]

    borders_interloctr = np.array([[0, len(bvh_interloctr) * HOP_LENGTH]])

    if len(borders_main) != 0:
        borders_interloctr = get_invert_borders(borders_main, len(audio_main))

    is_main_first = (borders_main[0][0] == 0) if len(borders_main) > 0 else False

    split_audio_main, split_gestures_main = get_split_data(borders_main, audio_main, bvh_interloctr)
    split_audio_interloctr, split_gestures_interloctr = get_split_data(borders_interloctr, audio_interloctr,
                                                                       bvh_interloctr)

    if len(split_audio_main) != 0 and len(split_audio_interloctr) != 0:
        assert np.concatenate(split_audio_main).shape[0] + np.concatenate(split_audio_interloctr).shape[0] == len(
            audio_main)
        assert np.concatenate(split_gestures_main).shape[0] + np.concatenate(split_gestures_interloctr).shape[0] == \
               bvh_interloctr.shape[0]
        assert np.concatenate(split_gestures_main).shape[1] == np.concatenate(split_gestures_interloctr).shape[1] == \
               bvh_interloctr.shape[1]

    split_audio_first, split_audio_second, split_gestures_first, split_gestures_second = get_split_by_order(
        split_audio_main, split_audio_interloctr, split_gestures_main, split_gestures_interloctr, is_main_first)

    gesture_blocks = []
    for i in range(len(split_audio_first)):
        beats_first = get_beats(split_audio_first[i], len(split_gestures_first[i]))
        bvh_blocks_first = split_bvh_into_blocks(split_gestures_first[i], beats_first)
        gesture_blocks += bvh_blocks_first

        if len(split_audio_second) > i:
            beats_second = get_beats(split_audio_second[i], len(split_gestures_second[i]))
            bvh_blocks_second = split_bvh_into_blocks(split_gestures_second[i], beats_second)
            gesture_blocks += bvh_blocks_second

    assert np.concatenate(gesture_blocks, axis=0).shape[0] == bvh_interloctr.shape[0]

    gest_len = np.array([len(g) for g in gesture_blocks])
    assert np.sum(gest_len) == bvh_interloctr.shape[0]
    return pad_gestures(gesture_blocks), gest_len


def save_vqvae_inference(src, dst, vqvae, is_train: bool):
    ensure_dir_exists(dst)
    codebook = vqvae.vqvae.vq.embedding.weight
    for audio_main_path in tqdm(Path(src).rglob('*.wav')):
        inter_gest_out, gest_len = get_gestures(str(audio_main_path))
        with torch.no_grad():
            encod_vector = vqvae.vqvae.encoder(inter_gest_out)
            vq_vect = vqvae.vqvae.vq(encod_vector, training=is_train)
            distances = torch.sum(encod_vector ** 2, dim=1, keepdim=True) + \
                        torch.sum(codebook ** 2, dim=1) - 2 * \
                        torch.matmul(encod_vector, codebook.t())

            codebook_ind = torch.argmin(distances, dim=-1).cpu().numpy()
            vq = codebook[codebook_ind].cpu().numpy()

        result = []

        for i in range(len(gest_len)):
            result += [vq[i]] * gest_len[i]

        result = np.array(result)
        dst_path = os.path.join(dst, audio_main_path.name.replace(".wav", ".npy"))
        np.save(str(dst_path), result)

    # deleting unused codes
    if is_train:
        unused_codes = vqvae.vqvae.get_unused_codes()
        print("Unused codes count: {}".format(len(unused_codes)))
        vqvae.vqvae.vq.reset_unused_codes_on_end(unused_codes)


def add_system_args(parent_parser: ArgumentParser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--src', type=str, default='data/trn/main-agent')
    parser.add_argument('--dst', type=str, default='vqvae_inf_data/trn')
    parser.add_argument('--checkpoint', type=str, help="Path to checkpoint")
    parser.add_argument('--train_data', action="store_true")
    parser.add_argument('--num_embeddings', type=int, default=2048)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=54)
    parser.add_argument('--hidden_dim', type=int, default=512)
    return parser


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser = add_system_args(arg_parser)
    args = arg_parser.parse_args()

    src = args.src
    dst = args.dst

    vqvae = VQVAESystem(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                        input_dim=args.input_dim, hidden_dim=args.hidden_dim, max_frames=MAX_BEATS_LEN)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    vqvae.load_state_dict(state_dict)
    vqvae.eval()

    save_vqvae_inference(src, dst, vqvae, is_train=args.train_data)

    if args.train_data:
        torch.save(vqvae.state_dict(), args.checkpoint)
