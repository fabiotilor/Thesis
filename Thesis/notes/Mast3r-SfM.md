# Paper Notes — DUSt3R
> **Full title:**
> **Authors:**
> **Venue / Year:**
> **Link:** https://arxiv.org/pdf/2409.19152

---

## 🧠 The Big Idea
*In 1–2 sentences, what is this paper actually doing? (write this in your own words)*

> Extending Mast3r to an actual SfM solution. Instead of only working on image pairs and then stitching them together, it decides itself on relevant view pairs.

---

## ❓ Problem Being Solved
*What is the gap or limitation in prior work that motivated this paper?*

>

---

## ⚙️ Method
*How do they solve it? Describe the pipeline / approach at a high level.*

> 

### Key components
- **Component 1:**
Scene Graph: To reduce number of images to check for matching due to quadratic complexity. G(V,E): V:= Images, E:= Two likely-overlapping images.
Image retrieval: Compute similarity scores of image pairs, that returns how likely it is that the images observe the same part of a scene. Encoder output turns image into set of local descriptors, use ASMK for image-retrieval.
- **Component 2:**
Local Reconstruction: Outputs local 3D reconstruction of image pairs. Mast3r directly predicts metric 3D structure 
- **Component 3:**
Refinement: Pointmaps are noisy due to ambiguities. Refine with global optimization on top of coarse estimation from Coarse alignment, which only optimizes scale and rigid pose. Now they optimize Depth maps, instrinsics, pose and scale, while minimizing 2D reprojection errors. The anchors are introduced to solve the problem that pairwise correspondences rarely form true multi-view point tracks, which makes refinement unstable if every pixel depth is optimized independently. They instead group pixels onto a coarse regular grid of anchor points and optimize depth only for these anchors, tying nearby pixels to the same anchor using relative depth ratios from the canonical depthmap. This greatly reduces the number of optimization variables while enforcing local geometric consistency, making global refinement more stable and efficient.


---

## 📐 Math / Formulation
*Any key equations worth noting. Write them here so you have to engage with them.*

>

---

## 📊 Experiments & Results
*What did they test on? What is the headline result?*

>

---

## 💡 Insights & Takeaways
*What did you find interesting, surprising, or elegant?*

>

---

## ❔ Questions & Confusions
*Things you didn't understand while reading. Come back to these.*

- [ ] 
- [ ] 
- [ ] 

---

## 🔍 Had to Google
*Concepts, terms, or methods you had to look up. Build your own mini-glossary.*

| Term | What it means |
|------|---------------|
|   frozen    |No longer updating model parameters               |
|      |               |
|      |               |

---

## 🔗 Related Work to Follow Up
*Papers mentioned that seem worth reading.*

- 
- 

---