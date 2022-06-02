Author: Sheng LinXiang

Email: getsheng6456@gmail.com

Current Affiliation: UESTC

Future Affiliation: University of Southern California

supervisor: Jin Qi

Date: 6/1/2022

note: this work was done while I pursued my bachelor degree in Jin Qi's AIML Lab in UESTC

# gpu-fingerprint-generator

This project is meant to generate master fingerprint images via simple numerical input.

You have to change the feature input in the main.cpp to use this generator.

CUDA environment is needed.

This project is based on [sfinge](https://github.com/zikohcth/sfinge) by [zikohcth](https://github.com/zikohcth) and [giapngvan](https://github.com/giapngvan).

# add-pores-and-scratched-for-master-fingerprint-image

This project is meant to add sweat pores and/or scratches for fingerprint image.

The input path and output path must be folders.

Command:

1. add_pore_scratch -i <folder path> -o <folder path>
2. add_pore_scratch -i <folder path> -o <folder path> -num <number> -dpi <number>

This executable file is **portable** and includes all the binaries and models required. No CUDA or PyTorch environment is needed.

Simple example of output can be found in the **output** folder.
