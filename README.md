# Amateur Football Analytics using Computer Vision
<p>
An application that accepts video input from soccer games, taken from TV perspective point of view and depicts 
teams on top view perspective.
Based primarily on existing work from papers [1], [2], current projects aims to apply existing methodologies for video 
input, with as less processing time per frame as possible.

![img.png](img.png)

Available demo on YouTube: https://www.youtube.com/watch?v=m6JKf9K2b44

### Instructions to run demo:
1. Download files DualPix2Pix models for court detection 
      - segmentation model weights: https://drive.google.com/file/d/1QCinahFH_830nH2RqwgoT8jehqxJgHQK/view?usp=share_link
      - line detection model weights: https://drive.google.com/file/d/1QzJzSUP9Zmqc4Eiko3dS1ZZTDmSQ0E10/view?usp=share_link
2. Place downloaded files in directory _/models/generated_models/two_pix2pix/_
3. Run main.py

## References
<a id="1">[1]</a> 
Komorowski, J., Kurzejamski, G., & Sarwas, G. (2019). Footandball: Integrated player and ball detector. arXiv preprint arXiv:1912.05445.

<a id="2">[2]</a> 
Chen, J., & Little, J. J. (2019). Sports camera calibration via synthetic data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 0-0).

---
## Citation
If you find this work useful in your research, please consider citing using the entry below:

@article{mavrogiannis2022amateur,
   title={Amateur football analytics using computer vision},
   author={Mavrogiannis, Panagiotis and Maglogiannis, Ilias},
   journal={Neural Computing and Applications},
   volume={34},
   number={22},
   pages={19639--19654},
   year={2022},
   publisher={Springer} 
}

---

Executable file:
* main.py

