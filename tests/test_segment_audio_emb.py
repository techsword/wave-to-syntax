"""Regression tests for the word-segmentation step in extract_segmented_embeddings.

The ``segment_audio_emb`` call was commented out in the tracked history, so
``generating_features`` stacked raw frame-level features (num_layers, T, 768)
instead of the published word-segmented schema (num_layers, n_words, 768).
These tests pin the restored behaviour with synthetic inputs. No corpora or GPU
are required.
"""

import pandas as pd
import torch

from spoken_syntax_probe.extract_segmented_embeddings import segment_audio_emb

HIDDEN = 768
TOTAL_FRAMES = 10
AUDIO_LEN = 1.0


def _segment_frame():
    # Columns before ``transcription`` match the alignment CSV layout that
    # ``segment_audio_emb`` expects (``iloc[:, 3:]`` must include
    # ``transcription`` plus the derived frame columns).
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "startTime": [0.0, 0.2, 0.5],
            "endTime": [0.2, 0.4, 1.0],
            "transcription": ["the", "sil", "cat"],
        }
    )


def _emb_layer(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, TOTAL_FRAMES, HIDDEN, generator=generator)


def test_segment_audio_emb_returns_one_row_per_word():
    emb = _emb_layer(0)
    segments = segment_audio_emb(emb, _segment_frame(), AUDIO_LEN)

    # Rows: "the" and "cat"; the "[sil]" row is filtered out.
    assert segments.shape == (2, HIDDEN)


def test_segment_audio_emb_averages_the_aligned_frames():
    emb = _emb_layer(1)
    segments = segment_audio_emb(emb, _segment_frame(), AUDIO_LEN)

    # startTime 0.0 -> frame 0, endTime 0.2 -> frame 2
    expected_the = emb[0, 0:2, :].mean(dim=0)
    # startTime 0.5 -> frame 5, endTime 1.0 -> frame 10
    expected_cat = emb[0, 5:10, :].mean(dim=0)

    assert torch.allclose(segments[0], expected_the)
    assert torch.allclose(segments[1], expected_cat)


def test_reused_segment_frame_across_layers_stays_word_level():
    # generating_features reuses one segment_df across all layers.
    n_words = 2
    segment_frame = _segment_frame()
    layers = [_emb_layer(seed) for seed in range(3)]

    segment_layers = [segment_audio_emb(x, segment_frame, AUDIO_LEN) for x in layers]
    stacked = torch.stack(segment_layers).detach().cpu().numpy()

    assert stacked.shape == (3, n_words, HIDDEN)
    # Word-segmented, not the raw frame-level stack.
    assert stacked.shape[1] != TOTAL_FRAMES
