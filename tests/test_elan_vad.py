from functools import reduce
from pathlib import Path, PosixPath
from typing import List

from pympi.Elan import Eaf
from elan_vad import __version__
from elan_vad.vad import (
    Annotation,
    add_vad_tier,
    cluster,
    cluster_tier_by_vad,
    combine_overlapping_annotations,
    detect_voice,
    Speech,
)

TEST_DATA_PATH = "data"


def test_version():
    assert __version__ == "0.2.0"


def test_annotation_join():
    a = Annotation(0, 1000, "hello")
    b = Annotation(500, 2000, "goodbye")

    def assert_join(annotation: Annotation):
        assert annotation.start == 0
        assert annotation.end == 2000
        assert annotation.annotation == "hello goodbye"

    assert_join(a.join(b))
    assert_join(b.join(a))


def test_annotation_overlap():
    a = Annotation(0, 1000, "hello")
    b = Annotation(1000, 2000, "goodbye")
    c = Annotation(500, 800, "goodbye")

    assert a.overlaps(b)
    assert b.overlaps(a)
    assert a.overlaps(c)
    assert not b.overlaps(c)


def test_annotation_append():
    a = Annotation(0, 1000, "hello")
    b = Annotation(1000, 2000, "goodbye")

    result = a.append_annotation(b)
    assert a.annotation == "hello"
    assert result.annotation == "hello goodbye"


def test_annotation_from_elan():
    elan = Eaf(Path(TEST_DATA_PATH, "cluster.eaf"))

    tier_id = "Phrase"
    start, end = 660, 2570
    value = "hekaai deina del ong hayei ba"

    raw_annotation = elan.get_annotation_data_for_tier(tier_id)[0]
    annotation = Annotation.from_elan_annotation(raw_annotation)
    assert annotation.annotation == value
    assert annotation.start == start
    assert annotation.end == end


def test_annotation_equals():
    a = Annotation(0, 1000, "hello")
    b = Annotation(0, 1000, "hello")
    c = Annotation(0, 2000, "goodbye")

    assert not a == 1
    assert a == a
    assert a == b
    assert b == a
    assert a != c


def test_detect_voice():
    test_audio = Path(TEST_DATA_PATH, "hello.wav")
    speech = detect_voice(test_audio)

    assert len(speech) > 0
    for section in speech:
        assert isinstance(section, Speech)


def test_add_vad_tier():
    elan_file = Path(TEST_DATA_PATH, "cluster.eaf")
    elan = Eaf(elan_file)

    voice_sections: List[Speech] = [
        Speech(start_ms=0, end_ms=1000),
        Speech(start_ms=2000, end_ms=3000),
    ]
    add_vad_tier(elan, voice_sections)
    vad_tier_data = elan.get_annotation_data_for_tier("_vad")
    assert len(vad_tier_data) == 2

    annotation_is_empty = lambda annotation: annotation[2] == ""
    assert all(map(annotation_is_empty, vad_tier_data))


def test_combine_overlapping_annotations():
    a = Annotation(start=0, end=1000, annotation="hello")
    b = Annotation(start=1200, end=1500, annotation="and")
    c = Annotation(start=1400, end=1800, annotation="goodbye")
    d = Annotation(start=1800, end=1900, annotation="weewoo")

    assert combine_overlapping_annotations([], a) == [a]
    assert combine_overlapping_annotations([a], b) == [a, b]
    assert combine_overlapping_annotations([a, b], c) == [a, b.join(c)]
    assert combine_overlapping_annotations([a, b.join(c)], d) == [a, b.join(c).join(d)]


def test_cluster():
    annotations = [
        Annotation(start=0, end=1000, annotation="hello"),
        Annotation(start=1200, end=1500, annotation="and"),
        Annotation(start=2000, end=2200, annotation="goodbye"),
        Annotation(start=2250, end=2600, annotation="weewoo"),
    ]
    vad_annotations = [
        Annotation(start=100, end=900, annotation=""),
        Annotation(start=1100, end=1600, annotation=""),
        Annotation(start=2000, end=2100, annotation=""),
        Annotation(start=2200, end=2300, annotation=""),
        Annotation(start=2400, end=2500, annotation=""),
    ]

    # Base case, empty result
    a = cluster([], annotations[0], vad_annotations)
    assert a == [vad_annotations[0].append_annotation(annotations[0])]

    # Add annotation to already existing annotation
    b = cluster(a, annotations[0], vad_annotations)
    assert b == [a[0].append_annotation(annotations[0])]

    # Add annotation completely contained within vad section
    c = cluster(a, annotations[1], vad_annotations)
    assert c == a + [vad_annotations[1].append_annotation(annotations[1])]

    # Add annotation which spans empty vad sections
    d = cluster(c, annotations[2], vad_annotations)
    # Should join the two vad sections and append the annotation
    assert d == c + [
        vad_annotations[2].join(vad_annotations[3].append_annotation(annotations[2]))
    ]

    # Add annotation which spans vad sections and overlaps with existing annotation
    e = cluster(d, annotations[3], vad_annotations)
    assert e == d[:-1] + [
        d[-1].join(vad_annotations[-1]).append_annotation(annotations[3])
    ]

    # Full reducer test
    reducer = lambda result, element: cluster(result, element, vad_annotations)
    assert reduce(reducer, annotations, []) == e


def test_full_workflow(tmp_path: PosixPath):
    sound_file = Path(TEST_DATA_PATH, "cluster.wav")
    elan_file = Path(TEST_DATA_PATH, "cluster.eaf")
    elan = Eaf(elan_file)
    original_annotation_data = elan.get_annotation_data_for_tier('Phrase')
    cluster_tier_id = 'vad_phrase'

    speech = detect_voice(sound_file)
    add_vad_tier(elan, speech)
    cluster_tier_by_vad(elan, annotation_tier_id='Phrase', vad_tier_id='_vad', cluster_tier_id=cluster_tier_id)

    # Check we haven't changed the original tier
    assert original_annotation_data == elan.get_annotation_data_for_tier('Phrase')

    # Check we've added the new tier
    assert cluster_tier_id in elan.get_tier_names()

    # Check the new tier has annotation data
    assert len(elan.get_annotation_data_for_tier(cluster_tier_id)) > 0

    # TODO remove after testing
    elan.to_file(Path(TEST_DATA_PATH, "test.eaf"))

    result_path = tmp_path.joinpath("cluster.eaf")
    elan.to_file(result_path)

    # Open and inspect result
    result = Eaf(result_path)
    assert "_vad" in result.get_tier_names()
    assert cluster_tier_id in result.get_tier_names()
