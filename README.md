# Thesis
Bachelor's Thesis Repo: Advancing Sparse Multi-View 4D Depth Estimation

**SiROP Link**
https://sirop.org/app/cad04039-4d69-48eb-98b9-95d2ac488803?_k=Lj6SAWPEm6htA1WA


Advancing Multi-View Depth Estimation Methods  Having 2-4 cameras instead of one can enable more precise and robust visual perception for robotics. The student will explore monocular and multi-view video depth estimators and work toward implementing a better method.
Keywords: deep learning, multi-view video depth estimation, dynamic reconstruction, 4D reconstruction
Description
Video depth estimation is a difficult and unsolved problem, especially in sparse multi-camera setups (e.g., four cameras). For videos, temporal stability can be difficult; e.g., directly combining per-frame depths often results in temporal jitter.
Having multiple (synchronized) cameras makes depth estimation even more difficult because the surfaces of objects need to align well from different cameras. With +100 cameras, this is not a problem and comercial products already exist, but when only 2-4 cameras are used, the setup becomes much more difficult and has not been well studied, although it offers promising advantages for numerous downstream applications, especially robotics where it can easily improve the visual perception quality due to the better, multi-view coverage of the scene.
In this project, the student will read papers to learn about monocular and multi-view video depth estimation methods, implement a simple evaluation setup for measuring reconstruction quality on 4-5 multi-view datasets (e.g., DexYCB [1], 4D-DRESS [2], Hi4D [3], Multi-View Kubric [4]), and then evaluate existing methods and work toward developing their own by combining existing approaches or implementing their own.
Starting points: MoGe [5], MegaSAM [6], MonoFusion [7], VGGT [8]. Other relevant papers: Easi3r [9], GeometryCrafter [10], Dust3r [11], GauSTAR [12], MapAnything [13], Dynamic 3DGS [14]. Seminal papers: MiDas [15].
Downstream application papers, for context: MVTracker [4], RoboTAP [16].
[1] DexYCB. In CVPR, 2021. https://dex-ycb.github.io.
[2] 4D-DRESS. In CVPR, 2024. https://eth-ait.github.io/4d-dress/.
[3] Hi4D. In CVPR, 2023. https://yifeiyin04.github.io/Hi4D/.
[4] MVTracker. In ICCV, 2025. https://ethz-vlg.github.io/mvtracker/.
[5] MoGe. In CVPR, 2025. https://wangrc.site/MoGePage/.
[6] MegaSAM. In CVPR, 2025. https://mega-sam.github.io.
[7] MonoFusion. In ICCV, 2025. https://github.com/Z1hanW/MonoFusion.
[8] VGGT. In CVPR, 2025. https://vgg-t.github.io.
[9] Easi3r. In ICCV, 2025. https://easi3r.github.io.
[10] GeometryCrafter. In ICCV, 2025. https://geometrycrafter.github
[11] Dust3r. In CVPR, 2024. https://github.com/naver/dust3r.
[12] GauSTAR. In CVPR, 2025. https://eth-ait.github.io/GauSTAR/.
[13] MapAnything. Arxiv, 2025. https://map-anything.github.io.
[14] Dynamic 3DGS. In 3DV, 2024. http://dynamic3dgaussians.github.io.
[15] MiDas. In TPAMI, 2020. https://github.com/isl-org/MiDaS.
[16] RoboTAP. In ICRA, 2024. https://robotap.github.io.

Goal

Learn about video depth estimation. Evaluate existing and implement your own method.
