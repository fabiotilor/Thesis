# Paper Notes — DUSt3R
> **Full title:**
> **Authors:**
> **Venue / Year:**
> **Link:** https://github.com/naver/mast3r

---

## 🧠 The Big Idea
*In 1–2 sentences, what is this paper actually doing? (write this in your own words)*

> Binocular. Changed the loss function to be metric, meaning the reconstruction actually gives us an accurate scale of the reconstructed object. Error is confidence-aware. Matching prediction via Matching head (2-layer MLP).

---

## ❓ Problem Being Solved
*What is the gap or limitation in prior work that motivated this paper?*

>

---

## ⚙️ Method
*How do they solve it? Describe the pipeline / approach at a high level.*

> Matching 3D point of one image to coordinate of the same point in the other image. Loss function L_match rewards matching of the correct pixel, not a nearby one -> High-precision matching. L_total = L_conf + Beta*L_match.

### Key components
- **Component 1:**
Fast matching, otherwise O(W^2H^2). We sample k points of a uniform grid and match only these, decreasing computation time but we do sacrifice completeness. For downstream tasks exhaustive correspondences is not needed. Due to this approach filtering out outliers, we get higher accuracies - remember we use regression as part of the Loss, which is sensitive to outliers.
- **Component 2:**
Matching on downscaled images, coarse-to-fine. This gives estimate of interesting regions. Next, split the full-sized (fine) image into 512px tiles and make them overlap 50%. Due to Coarse correspondences we know which window pairs are of interest to us. Then run mast3r in full resolution per window pair, giving us fine-grained feature maps. Repeat until 90% of coarse matches are covered. 
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
|      |               |
|      |               |
|      |               |

---

## 🔗 Related Work to Follow Up
*Papers mentioned that seem worth reading.*

- 
- 

---