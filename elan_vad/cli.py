from pathlib import Path
import click
from pympi.Elan import Eaf
from .vad import detect_voice, cluster_tier_by_vad, add_vad_tier


@click.command()
@click.argument("sound_file", type=click.Path(exists=True))
@click.argument("elan_file", type=click.Path(exists=True))
@click.option(
    "--vad-tier-id",
    default="_vad",
    type=str,
    help="The name of the tier to be created.",
)
def vad(sound_file: Path, elan_file: Path, vad_tier_id: str):
    click.echo(
        "{} in {}...".format(
            click.style("Detecting speech", bold=True),
            click.style(sound_file, bold=True, fg="green"),
        )
    )
    speech = detect_voice(sound_file)
    elan = Eaf(elan_file)

    click.echo(
        "{} to {}...".format(
            click.style("Writing VAD sections", bold=True),
            click.style(elan_file, bold=True, fg="green"),
        )
    )
    add_vad_tier(elan, speech, vad_tier_id)
    elan.to_file(elan_file)
    click.echo(
        click.style("Done!", bold=True),
    )


@click.command()
@click.argument("elan_file", type=click.Path(exists=True))
@click.argument("annotation_tier_id", default="_vad", type=str)
@click.option(
    "--vad-tier-id",
    default="_vad",
    type=str,
    help="The name of the vad-tier to use, defaults to _vad",
)
@click.option(
    "--cluster-tier-id",
    default="vad_cluster",
    type=str,
    help="The name of the cluster tier to be created. Defaults to vad_cluster",
)
def cluster(
    elan_file: Path, annotation_tier_id: str, vad_tier_id: str, cluster_tier_id: str
):
    elan = Eaf(elan_file)

    # Handle bad tier names
    for tier_id in (annotation_tier_id, vad_tier_id):
        if tier_id not in elan.get_tier_names():
            click.echo(
                "{}: {} not found in tier names for {}".format(
                    click.style(f"Invalid tier id", fg="red", bold=True),
                    click.style(tier_id, bold=True),
                    click.style(elan_file, bold=True),
                )
            )
            return

    click.echo(
        "{} annotations from tier: {}".format(
            click.style("Clustering", bold=True),
            click.style(annotation_tier_id, bold=True, fg="green"),
        )
    )
    cluster_tier_by_vad(elan, annotation_tier_id, vad_tier_id, cluster_tier_id)

    click.echo(
        "{} clustered annotations to tier: {}...".format(
            click.style("Writing", bold=True),
            click.style(cluster_tier_id, bold=True, fg="green"),
        )
    )
    elan.to_file(elan_file)
    click.echo(
        click.style("Done!", bold=True),
    )
