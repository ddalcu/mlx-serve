#!/usr/bin/env python3
"""Oracle for src/plugins/nemotron_transcribe/ctc_align.zig: NVIDIA's own aligner on synthetic CTC output.

  python3 tests/dump_ctc_align_fixtures.py <NeMo Speech checkout>   (needs numpy, scipy, torch)

Loads nemo/collections/speechlm2/parts/ctc_timestamp_utils.py with its NeMo imports stubbed (only the numpy/torch
alignment code runs), builds seeded scenarios (speaker streams, CTC log-probabilities with noise, Sortformer
activity) and runs `_align_banded_ctc` on them. Writes src/plugins/nemotron_transcribe/fixtures/ctc_align.json;
the Zig tests replay the inputs and demand the same word times and speaker mapping.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


def load_module(nemo_root: Path):
    stubs = {}
    for name in (
        "omegaconf",
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.modules",
        "nemo.collections.asr.modules.conv_asr",
        "nemo.collections.asr.modules.transformer_encoder",
        "nemo.collections.common",
        "nemo.collections.common.tokenizers",
        "nemo.collections.common.tokenizers.sentencepiece_tokenizer",
        "nemo.core",
        "nemo.core.utils",
        "nemo.core.utils.optional_libs",
    ):
        stubs[name] = types.ModuleType(name)
    stubs["omegaconf"].OmegaConf = object
    stubs["nemo.collections.asr.modules.conv_asr"].ConvASRDecoder = type("ConvASRDecoder", (torch.nn.Module,), {})
    stubs["nemo.collections.asr.modules.transformer_encoder"].TransformerEncoder = torch.nn.Module
    stubs["nemo.collections.common.tokenizers.sentencepiece_tokenizer"].SentencePieceTokenizer = object
    stubs["nemo.core.utils.optional_libs"].NUMBA_AVAILABLE = False
    sys.modules.update(stubs)
    path = nemo_root / "nemo/collections/speechlm2/parts/ctc_timestamp_utils.py"
    spec = importlib.util.spec_from_file_location("ctc_timestamp_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ctc_timestamp_utils"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


PARAMS = dict(
    speaker_activity_threshold=0.1881,
    speaker_logprob_weight=0.6057,
    maximum_token_len=1.0364,
    max_word_frames=38,  # 19 encoder frames at the head's x2 upsampling
    epsilon=1e-6,
)
CTC_STEP = 0.04


def scenario(rng, frames, speakers_in_talk, n_speaker_cols=4, vocab=7, words_per_turn=(2, 5), noise=1.5):
    """A transcript of turns by `speakers_in_talk` (tags), CTC peaks on a hidden path, Sortformer activity."""
    cols = vocab + 1  # column 0 = blank
    turns = []
    tag_cycle = list(speakers_in_talk)
    # Words: lists of token columns (1..vocab-1)
    budget = frames - 20
    used = 0
    wid = 0
    while True:
        tag = tag_cycle[len(turns) % len(tag_cycle)]
        words = []
        for _ in range(rng.integers(*words_per_turn)):
            n = int(rng.integers(1, 4))
            toks = [int(x) for x in rng.integers(1, cols, size=n)]
            need = sum(2 for _ in toks)
            if used + need * 2 > budget:
                break
            used += need * 2
            words.append(toks)
        if not words:
            break
        turns.append((tag, words))
        if used + 8 > budget:
            break
    # Lay the turns on the timeline: each token spans 1-2 frames then a blank gap.
    log_probs = np.full((frames, cols), -6.0, np.float32)
    log_probs[:, 0] = -0.05
    activity = np.zeros((frames, n_speaker_cols), np.float32)
    t = 4
    word_rows = []
    for tag, words in turns:
        col = tag % n_speaker_cols
        turn_start = t
        for toks in words:
            for tok in toks:
                width = int(rng.integers(1, 3))
                for k in range(width):
                    if t + k < frames:
                        log_probs[t + k, tok] = -0.1
                        log_probs[t + k, 0] = -3.0
                t += width + int(rng.integers(1, 4))
            t += int(rng.integers(0, 3))
        activity[max(0, turn_start - 2) : min(frames, t + 2), col] = 0.95
        t += int(rng.integers(2, 6))
    log_probs += rng.normal(0, noise * 0.05, log_probs.shape).astype(np.float32)
    log_probs -= np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
    activity = np.clip(activity + rng.normal(0, 0.03, activity.shape).astype(np.float32), 0, 1)
    flat = []
    for turn_index, (tag, words) in enumerate(turns):
        for toks in words:
            flat.append(dict(speaker_tag=tag, turn_index=turn_index, tokens=toks))
    return log_probs, activity, flat


def prepared_record(module, log_probs, activity, flat, with_speakers=True):
    frames, cols = log_probs.shape
    words = []
    for i, w in enumerate(flat):
        words.append(
            dict(word=f"w{i}", speaker_tag=w["speaker_tag"], turn_index=w["turn_index"], word_index=i, token_ids=list(w["tokens"]))
        )
    streams = []
    for tag, stream_words in module.MultiSpeakerSOTWordTimestampAligner._group_words_by_speaker(words).items():
        labels, state_to_word = module.MultiSpeakerSOTWordTimestampAligner._build_ctc_target(stream_words, 0)
        streams.append(dict(speaker_tag=tag, words=stream_words, labels=labels, state_to_word=state_to_word))
    # `_build_ctc_target` uses blank_id 0 here, so token ids already are the compact columns.
    rows = (frames + 1) // 2
    row_activity = torch.tensor(
        np.stack([activity[min(frames - 1, 2 * r) : min(frames, 2 * r + 2)].mean(axis=0) for r in range(rows)])
    )
    diar = np.repeat(activity, 4, axis=0) > 0.5  # 10 ms frames, 4 per CTC frame
    return dict(
        words=words,
        streams=streams,
        ctc_log_probs=torch.tensor(log_probs),
        ctc_length=frames,
        ctc_step=CTC_STEP,
        speaker_probs=row_activity if with_speakers else None,
        speaker_length=rows,
        sortformer_step=CTC_STEP * frames / rows,
        diarization_labels=torch.tensor(diar.T.copy()) if with_speakers else None,
        audio_duration=frames * CTC_STEP,
        time_offset=0.0,
    ), row_activity.numpy(), diar


def run(module, record):
    aligner = module.MultiSpeakerSOTWordTimestampAligner(
        speaker_activity_threshold=PARAMS["speaker_activity_threshold"],
        speaker_logprob_weight=PARAMS["speaker_logprob_weight"],
        maximum_token_len=PARAMS["maximum_token_len"],
        max_word_frames=PARAMS["max_word_frames"] // 2,
        epsilon=PARAMS["epsilon"],
    )
    prepared = dict(
        config=dict(PARAMS),
        num_speaker_columns=None if record["speaker_probs"] is None else record["speaker_probs"].shape[-1],
        diarization_frame_seconds=None if record["diarization_labels"] is None else 0.01,
        diarization_max_speaker_count=None if record["diarization_labels"] is None else record["diarization_labels"].shape[0],
        records=[record],
    )
    return aligner.align_prepared(prepared)[0]


def main():
    nemo_root = Path(sys.argv[1])
    module = load_module(nemo_root)
    cases = []
    for name, seed, frames, tags, spk in (
        ("one speaker, no activity", 1, 120, [0], False),
        ("two speakers", 2, 220, [0, 1], True),
        ("three speakers, tags out of order", 3, 300, [2, 0, 1], True),
        ("two speakers, long (banded)", 4, 1250, [0, 1], True),
        ("one speaker with activity", 5, 160, [1], True),
    ):
        rng = np.random.default_rng(seed)
        log_probs, activity, flat = scenario(rng, frames, tags)
        record, row_activity, diar = prepared_record(module, log_probs, activity, flat, with_speakers=spk)
        result = run(module, record)
        out_words = {}
        for tag, ws in result["speaker_word_timestamps"].items():
            for w in ws:
                out_words[int(w["word"][1:])] = dict(start=w["start"], end=w["end"])
        cases.append(
            dict(
                name=name,
                frames=frames,
                cols=int(log_probs.shape[1]),
                step=CTC_STEP,
                log_probs=[round(float(x), 6) for x in log_probs.ravel()],
                words=[dict(tag=w["speaker_tag"], tokens=w["tokens"]) for w in flat],
                speaker_rows=None if not spk else [round(float(x), 6) for x in row_activity.ravel()],
                speakers=int(activity.shape[1]),
                diar=None if not spk else [int(x) for x in diar.ravel()],
                diar_frames=int(diar.shape[0]),
                expected=[out_words[i] for i in range(len(flat))],
                mapping={str(k): v for k, v in result["speaker_tag_to_sortformer_column"].items()},
                failed=result["alignment_diagnostics"]["failed_streams"] != [],
            )
        )
        print(name, len(flat), "words", result["speaker_tag_to_sortformer_column"], result["word_timestamp_refinement"])
    dest = Path(__file__).resolve().parent.parent / "src/plugins/nemotron_transcribe/fixtures/ctc_align.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(dict(params=PARAMS, cases=cases)))
    print("wrote", dest, dest.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
