This code framework largely refers to **TorchCP**.  

This is because, before I wrote this framework, I carefully studied and memorized how **TorchCP** formally implements this function multiple times.

This framework supports thr, aps, saps, raps scores.

It could reproduce experiments of uncertainty-aware loss and c-adapter.

---

## Code Style

I try to follow suit google python code style and try to use object-oriented programming to improve scalability.

---

## Correctness

I checked the correctness of aps, saps and raps score by comparing with torchCP.
For the other parts, I try to avoid any bugs.

Hopefully, it is correct...