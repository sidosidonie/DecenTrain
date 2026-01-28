

## Flux.1-

- model: black-forest-labs/FLUX.1-schnell
- Step = 30
- A100 + 20 core CPU (google cloud)
- Verify Inference time: 403913.22 ms
- Normal Inference time: 117804.00 ms

```bash
Linear shapes: [input, weight, output] : num of calls
{(torch.Size([1, 256]), torch.Size([3072, 256]), torch.Size([1, 3072])): 30,
 (torch.Size([1, 512, 3072]), torch.Size([3072, 3072]), torch.Size([1, 512, 3072])): 2280,
 (torch.Size([1, 512, 3072]), torch.Size([12288, 3072]), torch.Size([1, 512, 12288])): 570,
 (torch.Size([1, 512, 4096]), torch.Size([3072, 4096]), torch.Size([1, 512, 3072])): 30,
 (torch.Size([1, 512, 12288]), torch.Size([3072, 12288]), torch.Size([1, 512, 3072])): 570,
 (torch.Size([1, 768]), torch.Size([3072, 768]), torch.Size([1, 3072])): 30,
 (torch.Size([1, 3072]), torch.Size([3072, 3072]), torch.Size([1, 3072])): 60,
 (torch.Size([1, 3072]), torch.Size([6144, 3072]), torch.Size([1, 6144])): 30,
 (torch.Size([1, 3072]), torch.Size([9216, 3072]), torch.Size([1, 9216])): 1140,
 (torch.Size([1, 3072]), torch.Size([18432, 3072]), torch.Size([1, 18432])): 1140,
 (torch.Size([1, 4096, 64]), torch.Size([3072, 64]), torch.Size([1, 4096, 3072])): 30,
 (torch.Size([1, 4096, 3072]), torch.Size([64, 3072]), torch.Size([1, 4096, 64])): 30,
 (torch.Size([1, 4096, 3072]), torch.Size([3072, 3072]), torch.Size([1, 4096, 3072])): 2280,
 (torch.Size([1, 4096, 3072]), torch.Size([12288, 3072]), torch.Size([1, 4096, 12288])): 570,
 (torch.Size([1, 4096, 12288]), torch.Size([3072, 12288]), torch.Size([1, 4096, 3072])): 570,
 (torch.Size([1, 4608, 3072]), torch.Size([3072, 3072]), torch.Size([1, 4608, 3072])): 3420,
 (torch.Size([1, 4608, 3072]), torch.Size([12288, 3072]), torch.Size([1, 4608, 12288])): 1140,
 (torch.Size([1, 4608, 15360]), torch.Size([3072, 15360]), torch.Size([1, 4608, 3072])): 1140,
 (torch.Size([1, 16384, 512]), torch.Size([512, 512]), torch.Size([1, 16384, 512])): 4}
 ```


 ## Product

 - Should we use quantized version? stable-diffusion.cpp?
 - Run on what type of GPU? 