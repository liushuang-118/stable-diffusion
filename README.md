
# 🎯 Object Location Guidance with Stable Diffusion

This project builds upon the **Stable Diffusion** framework to implement the **Object Location Guidance** method proposed in the paper [Universal Guidance for Diffusion Models (UGDM)](https://arxiv.org/abs/2302.07121). It extends the [Latent Diffusion Models (LDM)](https://arxiv.org/abs/2112.10752) framework to support object-level controllable generation and evaluation.

---

## 🔍 Cited Papers

### 🔧 Base Model: Latent Diffusion Models (LDM)

> Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
> **High-Resolution Image Synthesis with Latent Diffusion Models**, arXiv:2112.10752  
> [[arXiv Link]](https://arxiv.org/abs/2112.10752)

This project uses the LDM model as the base for image generation. We use the `sd-v1-4.ckpt` checkpoint.

### ✨ Control Method: Universal Guidance for Diffusion Models

> Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein  
> **Universal Guidance for Diffusion Models**, arXiv:2302.07121  
> [[arXiv Link]](https://arxiv.org/abs/2302.07121)

This project implements the **Object Location Guidance** control method from the paper, enabling precise control over object placement using text prompts, location hints, and forward optimization.

---

## 📦 Installation

### ✅ Install local project in editable mode (ensure `setup.py` exists)

```bash
pip install -e .
```


---

## 🧪 Example: Object Location Guided Generation

To run an object-location guided generation task:

```bash
mkdir -p test_od

python scripts/object_detection_with_evaluation.py \
  --indexes 0 \
  --text "a headshot of a woman with a dog" \
  --scale 1.5 \
  --optim_forward_guidance \
  --optim_num_steps 5 \
  --optim_forward_guidance_wt 100 \
  --optim_original_conditioning \
  --ddim_steps 50 \
  --optim_folder ./test_od/ \
  --ckpt /path/to/model.ckpt \
  --trials 3
```

**Argument Description:**

- `--text`: Text prompt describing the target image
- `--scale`: Classifier-free guidance scale
- `--optim_forward_guidance`: Enable location-based optimization guidance
- `--optim_num_steps`: Number of optimization steps
- `--optim_forward_guidance_wt`: Weight of optimization loss
- `--ddim_steps`: Number of DDIM sampling steps
- `--ckpt`: Path to Stable Diffusion `.ckpt` model
- `--optim_folder`: Output folder for results
- `--indexes`: Index(es) of images to test
- `--trials`: Number of generations per input

---

## 📂 Suggested Project Structure

```plaintext
project-root/
├── scripts/
│   └── object_detection_with_evaluation.py   # Main execution script
├── models/
│   └── sd-v1-4.ckpt                          # Stable Diffusion weights
├── taming-transformers/                     # Taming-transformers submodule
├── test_od/                                 # Output images folder
├── requirements.txt                         # Optional: dependency list
└── README.md                                # This file
```

---

## 📌 Notes

- This project focuses on inference and control, not training.
- Ensure `.ckpt` weights are legally obtained.
- GPU environment (CUDA 11.8) and PyTorch 2.0+ recommended.

---

## 📖 Citation

Please cite the following works if you use this project in your research:

```bibtex
@misc{rombach2021highresolution,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
  year={2021},
  eprint={2112.10752},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{bansal2023universalguidancediffusionmodels,
  title={Universal Guidance for Diffusion Models},
  author={Arpit Bansal and Hong-Min Chu and Avi Schwarzschild and Soumyadip Sengupta and Micah Goldblum and Jonas Geiping and Tom Goldstein},
  year={2023},
  eprint={2302.07121},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2302.07121}
}
```
