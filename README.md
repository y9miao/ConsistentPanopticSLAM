# Volumetric Semantically Consistent 3D Panoptic Mapping

**"Volumetric Semantically Consistent 3D Panoptic Mapping"** is a pipeline for incrementally building volumetric object-centric maps during online scanning with a localized RGB-D camera. The code framework is based on [**Voxblox++**](https://github.com/ethz-asl/voxblox-plusplus).
The main difference against **Voxblox++** is: 
<ol>
  <li> <a href="https://github.com/facebookresearch/detectron2">Panoptic segmentation</a> is applied in 2D RGB images instead of <a href="https://github.com/matterport/Mask_RCNN2">instance segmentation.</a></li>
  <li>A novel method to segment semantic-instance surface regions(super-points), as illustrated in Section III-B in the paper.</li>
  <li>A new graph optimization-based semantic labeling and instance refinement algorithm, as illustrated in Section III-C & Section III-D in the paper.</li>
  <li>The proposed framework achieves state-of-the-art 2D-to-3D instance segmentation accuracy, as illustrated in Section IV in the paper.</li>
</ol>

<p align="center">
  <img src="./images/pipeline.png" width=700>
</p>


## Getting started
- [Installing on Ubuntu](https://github.com/y9miao/volumetric-semantically-consistent-3D-panoptic-mapping/wiki/Installation)
- [Datasets](https://github.com/y9miao/volumetric-semantically-consistent-3D-panoptic-mapping/wiki/Datasets)
- [Basic usage](https://github.com/y9miao/volumetric-semantically-consistent-3D-panoptic-mapping/wiki/Basic-Usage)


## Citing
The framework is described in the following publication:

<!-- - Margarita Grinvald, Fadri Furrer, Tonci Novkovic, Jen Jen Chung, Cesar Cadena, Roland Siegwart, and Juan Nieto, **Volumetric Instance-Aware Semantic Mapping and 3D Object Discovery**, in _IEEE Robotics and Automation Letters_, July 2019. [[PDF](https://arxiv.org/abs/1903.00268)] [[Video](https://www.youtube.com/watch?v=Jvl42VJmYxg)]


```bibtex
@article{grinvald2019volumetric,
  author={M. {Grinvald} and F. {Furrer} and T. {Novkovic} and J. J. {Chung} and C. {Cadena} and R. {Siegwart} and J. {Nieto}},
  journal={IEEE Robotics and Automation Letters},
  title={{Volumetric Instance-Aware Semantic Mapping and 3D Object Discovery}},
  year={2019},
  volume={4},
  number={3},
  pages={3037-3044},
  doi={10.1109/LRA.2019.2923960},
  ISSN={2377-3766},
  month={July},
}
``` -->

If you use **Voxblox++** in your research, please cite accordingly.

## License
The code is available under the [BSD-3-Clause license](https://github.com/ethz-asl/voxblox-plusplus/blob/master/LICENSE).
