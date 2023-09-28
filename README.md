# Panoptic SLAM with semantic and geometric consistency

**"Panoptic SLAM with semantic and geometric consistency"** is a pipeline for simultaneously estimate image poses and incrementally building volumetric object-centric maps during online scanning with a localized RGB-D camera. The framework is divided into two parts: mapping and semantic-aided localization.

## Mapping  
The code framework is based on [**Voxblox++**](https://github.com/ethz-asl/voxblox-plusplus).
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

## Semantic-aided localization 

### Getting started
- [Installing on Ubuntu](https://github.com/y9miao/volumetric-semantically-consistent-3D-panoptic-mapping/wiki/Installation)
- [Datasets](https://github.com/y9miao/volumetric-semantically-consistent-3D-panoptic-mapping/wiki/Datasets)
- [Basic usage](https://github.com/y9miao/volumetric-semantically-consistent-3D-panoptic-mapping/wiki/Basic-Usage)

## TODO
- Update the code
- Integrate the graph-based optimization part into online mapping pipeline

## Citing
The framework is described in the following publication:

- Yang Miao, Iro Armeni, Marc Pollefeys, Daniel Barath, **Volumetric Semantically Consistent 3D Panoptic Mapping**, _arxiv_, 2023. [[PDF](https://arxiv.org/abs/2309.14737)] [[Video] - to upload]


```bibtex
@misc{miao2023volumetric,
      title={Volumetric Semantically Consistent 3D Panoptic Mapping}, 
      author={Yang Miao and Iro Armeni and Marc Pollefeys and Daniel Barath},
      year={2023},
      eprint={2309.14737},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

If you use our work in your research, please cite accordingly.
