<div align="center">
  <h1>Estimating 4D Data Associations Towards Spatial-Temporal Mapping of Growing Plants for Agricultural Robots</h1>
  <a href="https://github.com/PRBonn/plants_temporal_matcher#how-to-use-it"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/PRBonn/plants_temporal_matcher#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2023iros.pdf"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://lbesson.mit-license.org/"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>


<p>
  <i>This system can perform 3D point-to-point associations between plants' point clouds acquired in different session even in presence of highly repetitive structures and drastic changes.</i>
</p>

</div>



## Installation
First, clone our repository
```bash
git clone git@github.com:PRBonn/plants_temporal_matcher.git && cd plants_temporal_matcher
```

Then, we recommend setting up a virtual environment of your choice and installing the provided requirements through:
```bash
pip install -r requirements.txt
```



## How to Use It
We propose __two__ scripts:
* __temporal_matcher.py__ -> it compute associations between the point cloud extracted from a single frame and a reference map: (the script used to evaluate the system in our paper)
* __sparse_maps_matcher.py__ -> it takes two pre-computed maps and extract all the 3D point-to-point associations between them

In order to understand how to use the code it is important to keep in mind these __information__:
* The dataset is divided in __sessions__, each session is indicated by a number, ordered according to the time in which the recording was made
* We refer with the name __"reference"__ to the RGB-D sequence recorded first and with __"query"__ to the RGB-D sequence recorded after
* Each session is divided in __rows__, where each row is an actual different row in the glasshouse: of course, associations can be computed only between same rows


Type:
```bash
python temporal_matcher.py --help
```
or
```bash
python sparse_maps_matcher.py --help
```
to see how to run the scripts.
<details>
<summary>This is the output from the first script (<strong>temporal_matcher.py</strong>) </summary>

![temporal matcher help](https://github.com/PRBonn/plants_temporal_matcher/blob/main/images/temporal_matcher_help.png)

</details>
<details>
<summary>This is the output from the second script (<strong>sparse_maps_matcher.py</strong>) </summary>

![sparse maps matcher help](https://github.com/PRBonn/plants_temporal_matcher/blob/main/images/sparse_maps_matcher_help.png)

</details>

This is an __example__ on how to call the script:
```bash
python temporal_matcher.py /path/to/the/dataset/ --ref-number 1 --query-number 2 --row-number 3 --render-matches --no-visualize-map 
```



## Dataset
If you want to test this code on the dataset presented in the paper and reproducing the results, please send an email to [Luca Lobefaro](mailto:llobefar@uni-bonn.de?subject=[GitHub]%20Data%20Request).



## Publication
If you use our code in your academic work, please cite the corresponding [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/lobefaro2023iros.pdf):

```bibtex
@inproceedings{lobefaro2023iros,
  author = {L. Lobefaro and M.V.R. Malladi and O. Vysotska and T. Guadagnino and C. Stachniss},
  title = {{Estimating 4D Data Associations Towards Spatial-Temporal Mapping of Growing Plants for Agricultural Robots}},
  booktitle = iros,
  year = 2023,
  codeurl = {https://github.com/PRBonn/plants_temporal_matcher}
}
```



## License
This project is free software made available under the MIT License. For details see the [LICENSE](https://github.com/PRBonn/plants_temporal_matcher/blob/main/LICENSE) file.
