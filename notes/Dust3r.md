# Paper Notes — DUSt3R
> **Full title:**
> **Authors:**
> **Venue / Year:**
> **Link:** https://github.com/naver/dust3r

---

## 🧠 The Big Idea
*In 1–2 sentences, what is this paper actually doing? (write this in your own words)*

> Estimates 3D geometry as a learning problem from image pairs instead of: estimating camera intrinsics,
estimating camera extrinsics,
computing depth,
triangulating and
optimizing via bundle adjustment

---

## ❓ Problem Being Solved
*What is the gap or limitation in prior work that motivated this paper?*

>

---

## ⚙️ Method
*How do they solve it? Describe the pipeline / approach at a high level.*

> Encode the input images seperately using Transformers, that share their weights. Apply Self-Attention in this step. In the decoder apply cross-attention between the two distinct views. The seperate Regression Heads output the Pointmaps and the confidence scores. Views that are missing have NaN values, thus resulting in patchy 3D reconstruction.

### Key components
- **Component 1:**
Confidence score in loss (confidence of depth value prediction), enforces network to extrapolate in unconfident areas (e.g. single view).
- **Component 2:**
Global Alignment, for each image pair learn (optimization) best scale, rotation and translation to stitch the image pairs together for all image pairs. Thre transformations are made into a world-coordinate system
- **Component 3:**

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
| Camera intrinsics      |  Focal length in pixels, principal points and skew of camera. Mapping of 3D camera coordinates to image pixels             |
|   Camera extrinsics   |Cameras Location and Orientation               |
|      |               |

---

## 🔗 Related Work to Follow Up
*Papers mentioned that seem worth reading.*

- 
- 

---