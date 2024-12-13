from typing import cast
import numpy as np
import csv
from examples.local_generate import get_sample_protein
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
    InverseFoldingConfig,
)
from esm.sdk.forge import SequenceStructureForgeInferenceClient

def convert_none_to_nan(data):
    """Recursively convert None values in any deeply nested structure (e.g., list of lists of lists) to np.nan."""
    if isinstance(data, list):
        return [convert_none_to_nan(x) for x in data]
    elif data is None:
        return np.nan
    else:
        return data

def save_coordinates_as_pdb(protein: ESMProtein, filename: str):
    """Save protein coordinates to a .pdb file."""
    with open(filename, "w") as pdb_file:
        pdb_file.write("HEADER    Generated Protein Coordinates\n")
        for i, coord in enumerate(protein.coordinates):
            x, y, z = coord
            pdb_file.write(f"ATOM  {i+1:5d}  CA  ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
    print(f"Coordinates saved to {filename}")

def save_sequence_as_csv(protein: ESMProtein, filename: str):
    """Save protein sequence to a .csv file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence"])
        writer.writerow([protein.sequence])
    print(f"Sequence saved to {filename}")

def main(
    sequence_structure_client: SequenceStructureForgeInferenceClient,
    esm3_client: ESM3InferenceClient,
):
    # Folding with esm3 client
    protein = get_sample_protein()
    protein.coordinates = None
    protein.function_annotations = None
    protein.sasa = None
    assert protein.sequence is not None, "Protein sequence must be set to fold"

    # Folding with esm3 client
    config = GenerationConfig(track="structure", num_steps=1, temperature=0)
    esm3_client_folded_protein = esm3_client.generate(protein, config)
    assert isinstance(
        esm3_client_folded_protein, ESMProtein
    ), f"Using ESM3 client, ESMProtein was expected but got {protein}"

    # Save folding results
    save_coordinates_as_pdb(esm3_client_folded_protein, "esm3_folding_output.pdb")

    # Folding with folding client
    sequence_structure_client_folded_protein = sequence_structure_client.fold(
        protein.sequence, potential_sequence_of_concern=False
    )
    assert isinstance(
        sequence_structure_client_folded_protein, ESMProtein
    ), f"Using sequence_structure client, ESMProtein was expected but got {sequence_structure_client_folded_protein}"

    save_coordinates_as_pdb(sequence_structure_client_folded_protein, "sequence_structure_folding_output.pdb")

    # Inverse Folding with esm3 client
    protein = get_sample_protein()
    protein.sequence = None
    protein.sasa = None
    protein.function_annotations = None
    assert (
        protein.coordinates is not None
    ), "Protein coordinates must be set to inverse fold"
    config = GenerationConfig("sequence", num_steps=1, temperature=0.7)
    esm3_client_inv_folded_protein = cast(
        ESMProtein, esm3_client.generate(protein, config)
    )
    assert isinstance(
        esm3_client_inv_folded_protein, ESMProtein
    ), f"Using ESM3 client, ESMProtein was expected but got {protein}"

    # Save inverse folding results
    save_sequence_as_csv(esm3_client_inv_folded_protein, "esm3_inverse_folding_output.csv")

    # Inverse Folding with inverse folding client
    sequence_structure_client_inv_folded_protein = (
        sequence_structure_client.inverse_fold(
            protein.coordinates,
            config=InverseFoldingConfig(temperature=0.7),
            potential_sequence_of_concern=False,
        )
    )
    assert isinstance(
        sequence_structure_client_inv_folded_protein, ESMProtein
    ), f"Using sequence_structure client, ESMProtein was expected but got {sequence_structure_client_inv_folded_protein}"

    save_sequence_as_csv(sequence_structure_client_inv_folded_protein, "sequence_structure_inverse_folding_output.csv")

# Note: Ensure `sequence_structure_client` and `esm3_client` are properly instantiated before calling `main`. Call this in your main program:
# main(sequence_structure_client, esm3_client)
