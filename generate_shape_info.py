#!/usr/bin/env python3
#
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  Johns Hopkins Univ. (authors: Desh Raj)
#
# See ../LICENSE for clarification regarding multiple authors
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script takes the following two files as input:

    - cuts_train-clean-100.json.gz
    - bpe.model

to generate the shape information for benchmarking. The cut set should
additionally contain word-level alignment information in the supervisions,
if you want to benchmark the Alignment Restricted transducer.

The above two files can be generate by
https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh

The generated shape information is used to set the shape of randomly generated
data during benchmarking so that the benchmarking results look more realistic.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import sentencepiece as spm
import torch
from lhotse import load_manifest

DEFAULT_MAINIFEST = "/ceph-fj/fangjun/open-source-2/icefall-multi-datasets/egs/librispeech/ASR/data/fbank/cuts_train-clean-100.json.gz"  # noqa
DEFAULT_BPE_MODEL_FILE = "/ceph-fj/fangjun/open-source-2/icefall-multi-datasets/egs/librispeech/ASR/data/lang_bpe_500/bpe.model"  # noqa


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MAINIFEST,
        help="""Path to `cuts_train-clean-100.json.gz.
        It can be generated using
        https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=Path,
        default=DEFAULT_BPE_MODEL_FILE,
        help="""Path to the BPE model.
        It can be generated using
        https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh
        """,
    )

    parser.add_argument(
        "--generate-valid-ranges",
        action="store_true",
        help="""If true, generate the valid ranges for the
        Alignment Restricted transducer.""",
    )

    parser.add_argument(
        "--left-buffer",
        type=int,
        default=0,
        help="""Left buffer for the emission time of a token.""",
    )

    parser.add_argument(
        "--right-buffer",
        type=int,
        default=15,
        help="""Right buffer for the emission time of a token.""",
    )

    return parser


def encode_bpe_with_alignment(sp, alignment, frame_shift=0.01):
    """
    Encode a sentence using BPE with word-level alignment information. Propagate the
    alignment information to the BPE tokens. The output contains a list of tuples,
    where each tuple contains the BPE token and its corresponding alignment information
    in the format of (start_frame, end_frame).
    """
    bpe_alignments = []
    alignment = sorted(alignment, key=lambda x: x.start)
    for item in alignment:
        if item.symbol == "":
            continue
        bpe_tokens = sp.encode(item.symbol)
        duration_per_token = item.duration / len(bpe_tokens)
        for i, token in enumerate(bpe_tokens):
            st = (item.start + i * duration_per_token) / frame_shift
            en = (item.start + (i + 1) * duration_per_token) / frame_shift
            bpe_alignments.append((token, st, en))
    return bpe_alignments


def main():
    args = get_parser().parse_args()
    assert args.manifest.is_file(), f"{args.manifest} does not exist"
    assert args.bpe_model.is_file(), f"{args.bpe_model} does not exist"

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    cuts = load_manifest(args.manifest)

    TU_list = []
    if args.generate_valid_ranges:
        valid_ranges_list = []

    for c in tqdm(cuts, desc="Processing cuts"):
        sup = c.supervisions[0]
        num_frames = c.features.num_frames

        if args.generate_valid_ranges:
            if sup.alignment is None:
                continue
            tokens = encode_bpe_with_alignment(
                sp, sup.alignment["word"], frame_shift=c.features.frame_shift
            )

            # Compute the valid ranges for each token
            valid_ranges = torch.zeros(U, 2, dtype=torch.int32)
            for j, (token, st, en) in enumerate(tokens):
                valid_ranges[j, 0] = max(0, int(st - args.left_buffer))
                valid_ranges[j, 1] = min(num_frames, int(en + args.right_buffer))

            valid_ranges_list.append(valid_ranges)

        else:
            tokens = sp.encode(sup.text)

        U = len(tokens)

        # We assume the encoder has a subsampling_factor 4
        T = ((num_frames - 1) // 2 - 1) // 2
        TU_list.append([T, U])

    # NT_tensor has two columns.
    # column 0 - T
    # column 1 - U
    TU_tensor = torch.tensor(TU_list, dtype=torch.int32)
    print("TU_tensor.shape", TU_tensor.shape)
    torch.save(TU_tensor, "./shape_info.pt")
    print("Generate ./shape_info.pt successfully")

    if args.generate_valid_ranges:
        # valid_ranges_list contains a list of tensors. Each tensor has two columns.
        # column 0 - start frame
        # column 1 - end frame
        # Each row corresponds to a BPE token.
        torch.save(valid_ranges_list, "./valid_ranges.pt")
        print("Generate ./valid_ranges.pt successfully")


if __name__ == "__main__":
    main()
