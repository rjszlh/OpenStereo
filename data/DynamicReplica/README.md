# Prepare DynamicReplica Dataset

DynamicReplica consists of 145200 stereo frames (524 videos) with humans and animals in motion.

Dataset can be downloaded at the following website: https://github.com/facebookresearch/dynamic_stereo

The directory structure should be:
```text
DynamicReplica
└───disparity
|   └───train
|   └───valid
└───real
└───test
└───train
└───valid
    
```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{karaev2023dynamicstereo,
  title={DynamicStereo: Consistent Dynamic Depth from Stereo Videos},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={CVPR},
  year={2023}
}
```