## Notes and TODO

- [x] Set it up so that the code can be easily copied and run on colab, and download experiment output files.
- [x] Make sure it uses GPUs
- [ ] Make a small convnet and run experiment to get generalisation accuracy
- [ ] Find a standard benchmark convnet and run experiment to get generalisation accuracy
- [x] Implement more advanced coordinate embeddings in Coordinet architecture 
- [ ] Test different variations of Coordinet architecture
  - [ ] Whether it's better to more layers before or after embedding?
  - [ ] How big layers need to be?
- [ ] Run experiments with {Convnet, fully connected and Coordinet} to compare robustness to test dataset transformations
  - [ ] E.g. randomly shift test dataset images, then check test accuracy of those trained networks
  - [ ] Try to keep each model at roughly the same number of parameters (Might be hard with convnet?)

