# md-residence-time
Molecular Dynamics residence time calculation script

## Installations
1. To install argparse:
   ```console
   $ pip install --upgrade argparse
   ```
1. To install sklearn:
   ```console
   $ pip install --upgrade sklearn
   ```
1. To install tqdm:
   ```console
   $ pip install --upgrade tqdm
   ```

## Running the script
1. To get help about running the script:
    ```console
    $ python residence-time.py --help
    ```
1. To run the script:
    ```console
    $ python residence-time.py "path_to_data_file" "path_to_dump_file" adsorbent_atom_id_start adsorbent_atom_id_end adsorbate_atom_id_start adsorbate_atom_id_end
    ```
    For example,
    ```console
    $ python residence-time.py 500w_25h202_5pnp_sci_npt_100ps2.data 500w_25h202_5pnp_sci_npt_100ps2.dump 1 149 150 1649
    ```
