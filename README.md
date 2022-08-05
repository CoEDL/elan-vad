# Elan-vad 
Elan vad is a tool to perform Voice Activity Detection related tasks on Elan files

## Installation
You can install the package with `pip install elan-vad` (or `pip3` on macs).

After installation, you can import the utilities into your python program with:
```python
from elan_vad import *
```

The package additionally comes with two CLI programs: `vad` and `cluster`, which
can be used to perform the utilities from the terminal. 

## Usage
### As a Library
The example below: 
  - Performs VAD on an audio file, 
  - Adds these detected sections to an elan file (under the tier "\_vad"),
  - And then clusters the annotations within an existing tier ("Phrase") to be 
    constrained within the VAD sections.

```python
from pathlib import Path
from pympi.Elan import Eaf
from elan_vad import detect_voice, add_vad_tier, cluster_tier_by_vad

# Replace these paths with the correct values for your application
sound_file: Path = 'audio.wav'
elan_file: Path = 'test.eaf'

# Open up the Elan file for modification.
elan = Eaf(elan_file)

# Perform VAD on the sound_file
speech = detect_voice(sound_file)
add_vad_tier(elan, speech, '_vad')

# Cluster annotations within a 'Phrase' tier by the VAD sections
cluster_tier_by_vad(elan, 'Phrase', '_vad', 'vad_cluster')

# Replace the elan file with the new data
elan.to_file(elan_file)
```

### From the terminal
todo 
