import os
from pathlib import Path

import floris
from floris import FlorisModel


CONVERT_FOLDER = Path(__file__).resolve().parent / "v3_to_v4_convert_test"
FLORIS_FOLDER = Path(floris.__file__).resolve().parent


def test_v3_to_v4_convert():
    # Note certain filenames
    filename_v3_floris = "gch.yaml"
    filename_v4_floris = "gch_v4.yaml"
    filename_v3_turbine = "nrel_5MW_v3.yaml"
    filename_v4_turbine = "nrel_5MW_v3_v4.yaml"

    # Copy convert scripts from FLORIS_FOLDER to CONVERT_FOLDER
    os.system(f"cp {FLORIS_FOLDER / 'convert_turbine_v3_to_v4.py'} {CONVERT_FOLDER}")
    os.system(f"cp {FLORIS_FOLDER / 'convert_floris_input_v3_to_v4.py'} {CONVERT_FOLDER}")

    # Change directory to the test folder
    os.chdir(CONVERT_FOLDER)

    # Run the converter on the turbine file
    os.system(f"python convert_turbine_v3_to_v4.py {filename_v3_turbine}")

    # Run the converter on the floris file
    os.system(f"python convert_floris_input_v3_to_v4.py {filename_v3_floris}")

    # Go through the file filename_v4_floris and replace f"!include {filename_v3_turbine}"
    # with f"!include {filename_v4_turbine}"
    with open(filename_v4_floris, "r") as file:
        filedata = file.read()
    filedata = filedata.replace(
        f"!include {filename_v3_turbine}", f"!include {filename_v4_turbine}"
    )
    with open(filename_v4_floris, "w") as file:
        file.write(filedata)

    # Now confirm that the converted file can be loaded by FLORIS
    fmodel = FlorisModel(filename_v4_floris)

    # Now confirm this model runs
    fmodel.run()

    # Delete the newly created files to clean up
    os.system(f"rm {filename_v4_floris}")
    os.system(f"rm {filename_v4_turbine}")
    os.system("rm convert_turbine_v3_to_v4.py")
    os.system("rm convert_floris_input_v3_to_v4.py")
