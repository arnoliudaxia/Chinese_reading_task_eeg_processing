# Chinese Linguistic Corpus EEG Dataset Development and Advanced Semantic Decoding

## Introduction

This project aims to provide a comprehensive paradigm for the establishment of an EEG dataset based on Chinese linguistic corpus. It seeks to facilitate the advancement of technologies related to EEG-based semantic decoding and brain-computer interfaces. The project is currently divided into the following modules: Chinese corpus segmentation and text embeddings, experimental design and stimulus presentation, data preprocessing, and data masking. For detailed information on each module, please refer to the README document in the respective folders or view the relevant code.

For now, We recruited a total of 10 participants whose native language is Chinese. Each participant fully engaged in a Chinese novel reading task with a total duration of 12 hours, collectively accumulating 120 hours of data.

## Pipeline

Our EEG recording and pre-processing pipeline is as follows:

![](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/blob/main/image/pipeline_english.png)

## Device

### EEG Recording: EGI Geodesic EEG 400 series 

During the experiment, The EEG (electroencephalography) data were collected by a `128-channel` EEG system with Geodesic Sensor Net (EGI Inc., Eugene, OR, USA, [Geodesic EEG System 400 series (egi.com)](https://www.egi.com/clinical-division/clinical-division-clinical-products/ges-400-series)). The montage system of this device is `GSN-HydroCel-128`.We recorded the data at a sampling rate of 1000 Hz.

![](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/blob/main/image/egi_montage.png)

The 128-channel EEG system with Geodesic Sensor Net (GSN) by EGI is a sophisticated brain activity recording tool designed for high-resolution neuroscientific research. This system features an array of evenly spaced sensors providing complete scalp coverage, ensuring detailed spatial data collection without the need for interpolation. Coupled with the advanced Net Amps 400 amplifiers and intuitive Net Station 5 software, it delivers low noise, high sensitivity EEG data acquisition, and powerful data analysis capabilities, making it an ideal choice for dynamic and expanding research environments.

### Eyetracking: Tobii Pro Glasses 3

We utilized Tobii Pro Glasses 3 ([Tobii Pro Glasses 3 | Latest in wearable eye tracking - Tobii](https://www.tobii.com/products/eye-trackers/wearables/tobii-pro-glasses-3)) to record the participants' eye movement trajectories to inspect whether they followed the instructions of the experiment, that is, their gaze should move along with the red highlighted text.

The Tobii Pro Glasses 3 are advanced wearable eye trackers. They are capable of capturing natural viewing behavior in real-world environments, providing powerful insights from a first-person perspective. The device features 16 illuminators and four eye cameras integrated into scratch-resistant lenses, a wide-angle scene camera, and a built-in microphone, allowing for a comprehensive capture of participant behavior and environmental context. Its eye tracking is reliable across different populations, unaffected by eye color or shape. The Tobii Pro Glasses 3 operates with a high sampling rate of 50 Hz or 100 Hz. It supports a one-point calibration procedure. 

## Experiment

In the preparation phase of the experiment, we initially fitted participants with EEG caps and eye trackers, maintaining a distance of 67 cm from the screen. We emphasized to the participants that they should keep their heads still during the experiment, and their gaze should follow the red highlighted text as shown in the figure. 

![](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/blob/main/image/screen.png)

After ensuring the participants fully understood the instructions, we commenced the experimental procedure. Initially, there was an eye tracker calibration phase, followed by a practice reading phase, and finally the formal reading phase. Each formal reading phase lasted for approximately 30 minutes. The experimental setup is as below:

![](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/blob/main/image/exp_layout.png)

## Usage

Typically, you can follow steps below to execute the code for preparing experimental materials, conducting the experiment, and carrying out subsequent data analysis.

### Environment Settings

Firstly, please ensure that your code running environment is properly set up. You have the option to create a Docker container for this purpose or directly install the necessary packages on your personal computer. 

If you choose to use Docker, you can refer to the detailed tutorial provided [here](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/blob/main/docker/README.md). If you plan to install the packages in your local environment, the required packages and their corresponding version information can be found in the [requirement.txt](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/blob/main/requirements.txt) file located in the project's root directory.

### Experiment Materials Preparation

This step primarily involves preparing the textual reading materials needed for the experiment. You need to first convert your materials into the specific format below:

```
Chinese_novel.txt
Ch0
This is the preface of the novel
Ch1
Chapter 1 of the novel
Ch2
Chapter 2 of the novel
...
...
...
```

then run the `cut_Chinese_novel.py` script located in the `novel_segmentation` folder to perform sentence segmentation of the novel text:

```
python cut_Chinese_novel.py --divide_nums=<chapter numbers of the cutting point> --Chinese_novel_path=<path to your .txt file of the novel>
```

For detailed information on format requirements and script execution commands, please visit the [novel_segmentation_and_text_embeddings](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/tree/main/novel_segmentation_and_text_embeddings) module for more details.

### Experiment

Once we have obtained the text materials cut into the specific format, we can run the experimental program using `play_novel.py` in the `experiment` module. This program will present these text materials according to a specific experimental paradigm and record the participants' EEG and eye movement data. Before running the program, please ensure that the path to the text materials is correctly set and that the EEG and eye-tracking devices are properly connected. Use the following command to run the program:

```
python play_novel.py --add_mark --add_eyetracker  --preface_path=<your preface path> --host_IP=<host IP> --egi_IP=<egi IP> --eyetracker_hostname=<eyetracker serial number> --novel_path=<your novel path> --isFirstSession
```

For detailed information on the specific experimental paradigm, related parameter settings, and more, please refer to the [experiment](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/tree/main/experiment) module for further details.

### Data Pre-processing

After completing the experimental data collection for all participants, we can use the `preprocessing.py` in the `data_preprocessing` module for data preprocessing. Our preprocessing workflow includes a series of steps such as data segmentation, downsampling, filtering, bad channel interpolation, independent component analysis (ICA), and re-referencing. During the bad channel interpolation and ICA phases, we have implemented automated algorithms, but we also provide options for manual intervention to ensure accuracy. All parameters for these methods can be modified by adjusting the settings in the code. 

For detailed information on the preprocessing workflow, explanations of the code, and parameter settings, please refer to the [data_preprocessing](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/tree/main/data_preprocessing) module.

### Text Embeddings

We offer the embeddings of the reading materials. The text stimuli in each run has a corresponding embedding file saved in `.npy` format. These text embeddings provide a foundation for a series of subsequent studies, including the alignment analysis of EEG and textual data in the representation space, as well as tasks like EEG language decoding. For detailed information, please refer to the [novel_segmentation_and_text_embeddings](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/tree/main/novel_segmentation_and_text_embeddings) module.

### Data Alignment

After you have your texts, text embeddings and runs of EEG data, you can align them to do subsequent analysis. We offer you code to align the EEG data to its corresponding texts and embeddings. For detailed information, please refer to the [data_preprocessing_and_alignment](https://github.com/ncclabsustech/Chinese_reading_task_eeg_processing/tree/main/data_preprocessing_and_alignment) module.

## Credit 

- [Mou Xinyu](https://github.com/12485953) - Coder for all parts of the project, Data processing, README writer for all parts.

- [He Cuilin](https://github.com/CuilinHe) - Experiment conductor, Data processing, README writing.

- [Tan Liwei](https://github.com/tanliwei09) - Experiment conductor, Data processing, README writing.

- [Zhang Jianyu](https://github.com/ionaaaa) - Coder for Chinese corpus segmentation and EEG random masking.
  
- [Tian Yan](https://github.com/Bryantianyan) - Experiment conductor

- [Chen Yizhe]() - Experimental instrument debugging

  Feel free to contact us if you have any questions about the project !!!
  
## Collaborators
- [Wu Haiyan](https://github.com/haiyan0305)  -  澳门大学

- [Liu Quanying] - 南方科技大学
  
- [Wang Xindi](https://github.com/sandywang) 

- [Wang Qing] - 上海精神卫生中心
  
- [Chen Zijiao] - National University of Singapore
  
- [Yang Yu-Fang] - Freie Universität Berlin
  
- [Hu Chuanpeng] - 南京师范大学
  
- [Xu Ting] - Child Mind Institute

- [Cao Miao] - 北京大学

- [Liang Huadong](https://github.com/Romantic-Pumpkin) - 科大讯飞股份有限公司
## Funding

本项目受到天桥脑科学研究院MindX数据支持计划的部分资助（共计伍万元）。
其余资助来源:澳门科学技术发展基金项目(FDCT)
