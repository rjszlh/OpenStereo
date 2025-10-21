# Prepare FoundationStereo Dataset

FoundationStereo dataset is a large-scale (1M stereo pairs) synthetic training dataset featuring large diversity and high photorealism.
Dataset can be downloaded at the following website: https://drive.google.com/drive/folders/1YdC2a0_KTZ9xix_HyqNMPCrClpm0-XFU?usp=sharing

The directory structure should be:
```text
foundationstereo
└───amr_v5-b2_chaos_2500_1
|   └───dataset
|   |    └───data
|   |        └───left
|   |            └───rgb
|   |            └───disparity
|   |        └───right
|   |            └───rgb
...
```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{wen2025stereo,
  title={FoundationStereo: Zero-Shot Stereo Matching},
  author={Bowen Wen and Matthew Trepte and Joseph Aribido and Jan Kautz and Orazio Gallo and Stan Birchfield},
  journal={CVPR},
  year={2025}
}
```