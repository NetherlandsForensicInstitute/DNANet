"""Tests for inference output dataclasses."""

import json
from pathlib import Path

import pytest

from dnanet.infer.output import (
    AlleleCall,
    MarkerResult,
    ProfileResult,
    InferenceResult,
    save_epg_plot,
)


class TestAlleleCall:
    def test_creation(self):
        allele = AlleleCall(name='12', base_pair=120.5, height=1500.0, confidence=0.95)
        assert allele.name == '12'
        assert allele.base_pair == 120.5
        assert allele.height == 1500.0
        assert allele.confidence == 0.95

    def test_to_dict(self):
        allele = AlleleCall(name='13', base_pair=130.0, height=2000.0, confidence=0.88)
        d = {
            'name': allele.name,
            'base_pair': allele.base_pair,
            'height': allele.height,
            'confidence': allele.confidence,
        }
        assert d == {
            'name': '13',
            'base_pair': 130.0,
            'height': 2000.0,
            'confidence': 0.88,
        }

    def test_frozen(self):
        allele = AlleleCall(name='12', base_pair=120.0, height=1000.0, confidence=0.9)
        with pytest.raises(AttributeError):
            allele.name = 'changed'


class TestMarkerResult:
    def test_creation_empty(self):
        marker = MarkerResult(name='D3S1358', dye_row=0)
        assert marker.name == 'D3S1358'
        assert marker.dye_row == 0
        assert marker.alleles == []

    def test_creation_with_alleles(self):
        alleles = [
            AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95),
            AlleleCall(name='15', base_pair=132.0, height=2200.0, confidence=0.88),
        ]
        marker = MarkerResult(name='D3S1358', dye_row=0, alleles=alleles)
        assert len(marker.alleles) == 2
        assert marker.alleles[0].name == '12'

    def test_to_dict(self):
        alleles = [AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95)]
        marker = MarkerResult(name='D3S1358', dye_row=0, alleles=alleles)
        d = marker.to_dict()
        assert d['name'] == 'D3S1358'
        assert d['dye_row'] == 0
        assert len(d['alleles']) == 1
        assert d['alleles'][0]['name'] == '12'


class TestProfileResult:
    def test_creation_minimal(self):
        profile = ProfileResult(sample='sample1', hid_path='/data/sample1.HID')
        assert profile.sample == 'sample1'
        assert profile.hid_path == '/data/sample1.HID'
        assert profile.markers == []
        assert profile.warnings == []
        assert profile.allele_count == 0
        assert profile.marker_count == 0

    def test_creation_full(self):
        markers = [
            MarkerResult(
                name='D3S1358',
                dye_row=0,
                alleles=[
                    AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95),
                    AlleleCall(name='15', base_pair=132.0, height=2200.0, confidence=0.88),
                ],
            ),
            MarkerResult(
                name='vWA',
                dye_row=1,
                alleles=[AlleleCall(name='17', base_pair=200.0, height=1800.0, confidence=0.92)],
            ),
        ]
        profile = ProfileResult(
            sample='sample1',
            hid_path='/data/sample1.HID',
            ladder_path='/data/ladder.HID',
            markers=markers,
            warnings=['Low signal in dye 4'],
        )
        assert profile.allele_count == 3
        assert profile.marker_count == 2
        assert 'Low signal' in profile.warnings[0]

    def test_to_dict_excludes_none_optional_fields(self):
        profile = ProfileResult(sample='s1', hid_path='/data/s1.HID')
        d = profile.to_dict()
        assert 'ladder_path' not in d
        assert 'warnings' not in d
        assert d['sample'] == 's1'
        assert d['hid_path'] == '/data/s1.HID'
        assert d['markers'] == []

    def test_to_dict_includes_optional_fields_when_present(self):
        profile = ProfileResult(
            sample='s1',
            hid_path='/data/s1.HID',
            ladder_path='/data/ladder.HID',
            warnings=['warning1'],
        )
        d = profile.to_dict()
        assert d['ladder_path'] == '/data/ladder.HID'
        assert d['warnings'] == ['warning1']


class TestInferenceResult:
    def test_creation(self):
        result = InferenceResult(checkpoint='best.ckpt', kit='PPF6C')
        assert result.checkpoint == 'best.ckpt'
        assert result.kit == 'PPF6C'
        assert result.profiles == []
        assert result.to_dict()['total_profiles'] == 0
        assert result.to_dict()['total_alleles'] == 0

    def test_with_profiles(self):
        profiles = [
            ProfileResult(
                sample='s1',
                hid_path='/data/s1.HID',
                markers=[
                    MarkerResult(
                        name='D3S1358',
                        dye_row=0,
                        alleles=[
                            AlleleCall(name='12', base_pair=120.0, height=1500.0, confidence=0.95)
                        ],
                    ),
                ],
            ),
        ]
        result = InferenceResult(
            checkpoint='best.ckpt',
            kit='PPF6C',
            profiles=profiles,
            timing={'s1.HID': 0.5},
        )
        d = result.to_dict()
        assert d['checkpoint'] == 'best.ckpt'
        assert d['kit'] == 'PPF6C'
        assert d['total_profiles'] == 1
        assert d['total_alleles'] == 1
        assert 'timing' in d
        assert d['timing']['s1.HID'] == 0.5

    def test_save_json(self, tmp_path):
        result = InferenceResult(
            checkpoint='best.ckpt',
            kit='PPF6C',
            profiles=[
                ProfileResult(
                    sample='s1',
                    hid_path='/data/s1.HID',
                    markers=[
                        MarkerResult(
                            name='D3S1358',
                            dye_row=0,
                            alleles=[
                                AlleleCall(
                                    name='12', base_pair=120.0, height=1500.0, confidence=0.95
                                )
                            ],
                        ),
                    ],
                ),
            ],
        )
        output_path = tmp_path / 'results' / 'inference.json'
        saved = result.save_json(output_path)
        assert saved.exists()
        content = json.loads(saved.read_text())
        assert content['checkpoint'] == 'best.ckpt'
        assert content['kit'] == 'PPF6C'
        assert content['total_profiles'] == 1

    def test_total_markers_called(self):
        profiles = [
            ProfileResult(
                sample='s1',
                hid_path='/data/s1.HID',
                markers=[
                    MarkerResult(name='D3S1358', dye_row=0, alleles=[]),
                    MarkerResult(name='vWA', dye_row=1, alleles=[]),
                ],
            ),
            ProfileResult(
                sample='s2',
                hid_path='/data/s2.HID',
                markers=[MarkerResult(name='D3S1358', dye_row=0, alleles=[])],
            ),
        ]
        result = InferenceResult(checkpoint='best.ckpt', kit='PPF6C', profiles=profiles)
        assert result.total_markers_called == 3


class TestSaveEPGPlot:
    def test_save_epg_plot_creates_file(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng()
        signal = rng.random((5, 4096)).astype(np.float32) * 1000
        prediction = rng.random((5, 4096)).astype(np.float32)

        output_path = tmp_path / 'epg.png'
        result = save_epg_plot(
            signal=signal.tolist(),
            prediction=prediction.tolist(),
            title='test_profile',
            output_path=output_path,
        )
        assert result.exists()
        assert result.stat().st_size > 1000  # reasonable minimum PNG size

    def test_save_epg_plot_without_prediction(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng()
        signal = rng.random((5, 4096)).astype(np.float32) * 1000

        output_path = tmp_path / 'epg_no_pred.png'
        result = save_epg_plot(
            signal=signal.tolist(),
            title='test_profile',
            output_path=output_path,
        )
        assert result.exists()
