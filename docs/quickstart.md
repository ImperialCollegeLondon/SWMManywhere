# Quickstart

SWMManywhere is a Python tool to synthesise Urban Drainage Models (UDM) anywhere in the world.
It handles everything from data acquisition to running the UDM in the [SWMM](https://www.epa.gov/sites/default/files/2019-02/documents/epaswmm5_1_manual_master_8-2-15.pdf) software.

## Configuration

SWMManywhere is primarily designed to be used via a Command Line Interface (CLI).
The user provides a `config` file address that specifies a variety of options to customise the synthesis process.
However, the minimum requirements for a user to provide are simply:

- a base directory,
- a project name,
- a bounding box that specifies the latitude and longitude (EPSG:4326) of the bottom left and upper right corners of the region within which to create the UDM.

We can define a simple configuration `.yml` file here:

```yml
base_dir: /path/to/base/directory
project: my_first_swmm
bbox: [1.52740,42.50524,1.54273,42.51259]
```

## Run SWMManywhere

The basic command is:

```sh
python -m swmmanywhere --config_path=/path/to/file.yml
```

which will create a SWMM input file (`.inp`) at the file location:

```text
<base_dir>/<project>/bbox_1/model_1/model_1.inp
```

## Use your model

If you prefer GUIs then the easiest thing now is to download the [SWMM software](https://www.epa.gov/water-research/storm-water-management-model-swmm) and load your model in there.
The example above looks as follows:

![SWMM Model](images/andorra_swmm_screenshot.png)

From here you can run or edit your model.

If you want to investigate your model in GIS, then the geospatial data that was formatted into the model file (`model_1.inp`) is also available at:

```text
<base_dir>/<project>/bbox_1/model_1/nodes.geojson
<base_dir>/<project>/bbox_1/model_1/edges.geojson
<base_dir>/<project>/bbox_1/model_1/subcatchments.geojson
```

## Not happy with your model?

Then it sounds like you want to explore the wide range of customisability that SWMManywhere offers!
See our notebooks to understand what is going on in greater detail and how to create better synthetic UDMs.
