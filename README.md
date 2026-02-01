# Debugging GAN Exercises
Debugging GAN training loop – PyTorch project for NHL Stenden application

This repository contains my solutions for a series of debugging exercises
for the **Computer Vision & Data Science** course at **NHL Stenden University of Applied Sciences**.

The goal of the assignment is to identify and fix logical, structural and
cosmetic bugs in small pieces of Python code, and to document the
reasoning behind each fix.

---

## File overview

### `debug_exercises.py`

This file contains the corrected implementations for all four exercises:

#### Exercise 1 – `id_to_fruit`

- **Original issue**: iterating over a `set` and indexing by position.  
  Sets are unordered, so index–element mapping is not deterministic.
- **Fix**: convert the input to a deterministic sequence before indexing.
  In this solution `sorted(fruits)` is used to obtain a stable order.

#### Exercise 2 – `swap`

- **Task**: swap x and y coordinates of bounding boxes with rows in the
  form `[x1, y1, x2, y2, class_id]`.
- **Original issue**: incorrect assignments and in‑place modification of
  the same array during multiple assignment.
- **Fix**: work on a copy of the array and explicitly swap  
  `(x1, y1)` and `(x2, y2)` so that the result becomes  
  `[y1, x1, y2, x2, class_id]`.

#### Exercise 3 – `plot_data`

- **Task**: plot a precision–recall curve from values stored in a CSV file.
- **Original issue**: precision and recall were swapped on the axes
  (recall on x, precision on y), which contradicted the documentation.
- **Fix**: use precision on the x‑axis and recall on the y‑axis and label
  the axes accordingly. The plotting code now matches the CSV values.

#### Exercise 4 – GAN (`Generator`, `Discriminator`, `train_gan`)

- **Task**: debug a simple GAN training loop based on MNIST.
- **Structural issue**: label tensors were created with a fixed
  `batch_size`, which caused size mismatches for the last batch or when
  `batch_size` changed (e.g. from 32 to 64).
- **Fix**: use the **actual** batch size for each iteration  
  (`current_bs = real_samples.size(0)`) when creating labels and latent
  vectors. This guarantees that input and target sizes always match.
- **Cosmetic issue**: the progress display condition was based on
  `if n == batch_size - 1`, which depends on the number of batches and
  often never triggered.
- **Fix**: report progress in a clearer way (e.g. once per epoch using
  `if n == 0`) and print discriminator/generator losses.

---

## How to run

From a Python environment with PyTorch, NumPy, torchvision and Matplotlib:

```bash
python debug_exercises.py
