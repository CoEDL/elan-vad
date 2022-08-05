from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from functools import reduce
import pympi.Elan as Elan
import torch

SAMPLE_RATE = 16_000


@dataclass
class Speech:
    """A section containing detected speech."""

    start_ms: int
    end_ms: int

    def to_seconds(self) -> Tuple[int, int]:
        result = self.start_ms, self.end_ms
        to_seconds = lambda timestamp: round(timestamp * (1000 / SAMPLE_RATE))
        return tuple(map(to_seconds, result))


@dataclass
class Annotation:
    """A section of speech containing an annotation"""

    start: int
    end: int
    annotation: str

    def join(self, annotation: "Annotation") -> "Annotation":
        if self.start <= annotation.start:
            order = self, annotation
        else:
            order = annotation, self

        combined_annotation = " ".join(map(lambda x: x.annotation, order)).strip()
        return Annotation(
            start=min(self.start, annotation.start),
            end=max(self.end, annotation.end),
            annotation=combined_annotation,
        )

    def append_annotation(self, annotation: "Annotation") -> "Annotation":
        combined_annotation = f"{self.annotation} {annotation.annotation}".strip()
        return Annotation(
            start=self.start, end=self.end, annotation=combined_annotation
        )

    def overlaps(self, annotation: "Annotation") -> bool:
        return max(self.start, annotation.start) <= min(self.end, annotation.end)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Annotation):
            return False

        annotation = Annotation(__o.start, __o.end, __o.annotation)
        return (
            annotation.start == self.start
            and annotation.end == self.end
            and annotation.annotation == self.annotation
        )

    @staticmethod
    def from_elan_annotation(elan_annotation: Tuple[float, float, str]) -> "Annotation":
        start, end, annotation = elan_annotation
        return Annotation(round(start), round(end), annotation)


def detect_voice(sound_file: Path) -> List[Speech]:
    """Return a list of detected speech sections in the supplied adio file.

    Parameters:
        sounds_file: A path to a .wav file.
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, read_audio, *_) = utils

    audio = read_audio(sound_file, sampling_rate=SAMPLE_RATE)
    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=SAMPLE_RATE)

    return [
        Speech(start_ms=timestamp["start"], end_ms=timestamp["end"])
        for timestamp in speech_timestamps
    ]


def add_vad_tier(
    elan: Elan.Eaf, voice_sections: List[Speech], vad_tier_id: str = "_vad"
) -> None:
    """Adds a VAD tier to a supplied elan file, containing an empty annotation
    for each section of detected speech.

    Parameters:
        elan: An Eaf object in which to add the tier.
        voice_sections: Speech sections representing detected speech.
        vad_tier_id: The name of the VAD tier.
    """
    elan.add_tier(vad_tier_id)
    for section in voice_sections:
        start, end = section.to_seconds()
        elan.add_annotation(vad_tier_id, start=start, end=end, value="")


def add_annotations(
    elan: Elan.Eaf, annotations: List[Annotation], tier_id: str
) -> None:
    """Creates a tier within the elan file containing the supplied annotation
    data.

    This overwrites any data contained in the original tier.

    Parameters:
        elan: The eaf object to modify.
        annotations: A list of annotations to insert within the tier.
        tier_id: The name of the tier to create/overwrite
    """
    elan.add_tier(tier_id)
    for annotation in annotations:
        elan.add_annotation(
            tier_id,
            start=annotation.start,
            end=annotation.end,
            value=annotation.annotation,
        )


def cluster_tier_by_vad(
    elan: Elan.Eaf, annotation_tier_id: str, vad_tier_id: str, cluster_tier_id: str
) -> None:
    """Clusters the annotations within a tier by a corresponding VAD tier, and
    writes the combined annotations to a new tier.

    This groups utterances that occur within a section of speech. Utterances
    which span multiple VAD sections result in those sections being combined.

    Parameters:
        elan: The eaf file to modify
        annotation_tier_id: The name of the tier containing the annotations to cluster
        vad_tier_id: The name of the tier containing VAD sections
        cluster_tier_id: The name of the new tier containing annotation clusters.
    """
    # Extract and combine overlapping annotations within tier
    annotations = [
        Annotation.from_elan_annotation(annotation)
        for annotation in elan.get_annotation_data_for_tier(annotation_tier_id)
    ]
    annotations = reduce(combine_overlapping_annotations, annotations, [])
    # Restrict annotations to vad sections
    vad_annotations = [
        Annotation.from_elan_annotation(annotation)
        for annotation in elan.get_annotation_data_for_tier(vad_tier_id)
    ]

    # Closure to include the vad sections in the reducer
    cluster_reducer = lambda result, element: cluster(result, element, vad_annotations)
    result = reduce(cluster_reducer, annotations, [])
    add_annotations(elan=elan, annotations=result, tier_id=cluster_tier_id)


def combine_overlapping_annotations(
    annotations: List[Annotation], next_annotation: Annotation
) -> List[Annotation]:
    """Adds an annotation to the list, joining it with the last annotation in
    the list if the new annotation overlaps with it.

    This assumes the precondition that the next annotation can only overlap with
    at maximum one of the annotations in the list, and that the next_annotation
    must have an end time which is greater than the end time of the last annotation.

    Parameters:
        annotations: A list of non-overlapping annotations in ascending order of
            start and end times.
        next_annotation: The annotation to add to the list.

    Returns:
        A list of non-overlapping annotations which includes either the next_annotation,
            or the next_annotation joined with the last annotation.
    """
    if len(annotations) == 0:
        return [next_annotation]
    if next_annotation.overlaps(annotations[-1]):
        return [*annotations[:-1], annotations[-1].join(next_annotation)]
    return annotations + [next_annotation]


def cluster(
    result: List[Annotation], element: Annotation, vad_annotations: List[Annotation]
) -> List[Annotation]:
    vad_overlaps = [
        annotation for annotation in vad_annotations if element.overlaps(annotation)
    ]
    if len(vad_overlaps) == 0:
        return result

    annotation_overlaps = [section for section in result if element.overlaps(section)]
    if len(vad_overlaps) == 1:
        # Do we have a current annotation for that vad section?
        if len(annotation_overlaps) == 0:
            return result + [vad_overlaps[0].append_annotation(element)]
        else:
            # Note: Can only be one annotation overlap in this case
            annotation = annotation_overlaps[0]
            return result[:-1] + [annotation.append_annotation(element)]

    # Multiple VAD overlaps
    if len(annotation_overlaps) == 0:
        joined_vad_sections = reduce(
            Annotation.join,
            vad_overlaps,
        )
        return result + [joined_vad_sections.append_annotation(element)]

    # Can only be one annotation, join it with the last vad section.
    annotation = annotation_overlaps[0]
    spanning_annotation = annotation.join(vad_overlaps[-1])
    spanning_annotation = spanning_annotation.append_annotation(element)
    return result[:-1] + [spanning_annotation]
