

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

## Z-image

- Short prompt
 - Origin time for 1 step: 6958.60 ms
 - Verified time for 1 step: 15948.18 ms

 - Matmul shapes
```
Inference time: 15948.18 ms
{(torch.Size([1, 16384, 512]), torch.Size([512, 512]), torch.Size([1, 16384, 512])): 4,
 (torch.Size([2, 32, 3840]), torch.Size([3840, 3840]), torch.Size([2, 32, 3840])): 8,
 (torch.Size([2, 32, 3840]), torch.Size([10240, 3840]), torch.Size([2, 32, 10240])): 4,
 (torch.Size([2, 32, 10240]), torch.Size([3840, 10240]), torch.Size([2, 32, 3840])): 2,
 (torch.Size([2, 256]), torch.Size([1024, 256]), torch.Size([2, 1024])): 1,
 (torch.Size([2, 256]), torch.Size([3840, 256]), torch.Size([2, 3840])): 1,
 (torch.Size([2, 256]), torch.Size([15360, 256]), torch.Size([2, 15360])): 32,
 (torch.Size([2, 1024]), torch.Size([256, 1024]), torch.Size([2, 256])): 1,
 (torch.Size([2, 4096, 3840]), torch.Size([3840, 3840]), torch.Size([2, 4096, 3840])): 8,
 (torch.Size([2, 4096, 3840]), torch.Size([10240, 3840]), torch.Size([2, 4096, 10240])): 4,
 (torch.Size([2, 4096, 10240]), torch.Size([3840, 10240]), torch.Size([2, 4096, 3840])): 2,
 (torch.Size([2, 4128, 3840]), torch.Size([64, 3840]), torch.Size([2, 4128, 64])): 1,
 (torch.Size([2, 4128, 3840]), torch.Size([3840, 3840]), torch.Size([2, 4128, 3840])): 120,
 (torch.Size([2, 4128, 3840]), torch.Size([10240, 3840]), torch.Size([2, 4128, 10240])): 60,
 (torch.Size([2, 4128, 10240]), torch.Size([3840, 10240]), torch.Size([2, 4128, 3840])): 30,
 (torch.Size([64, 2560]), torch.Size([3840, 2560]), torch.Size([64, 3840])): 1,
 (torch.Size([8192, 64]), torch.Size([3840, 64]), torch.Size([8192, 3840])): 1}
```
- Longer prompt
 - Origin time for 1 step: 7059.03 ms 
 - Verified time for 1 step: 16114.97 ms
 - linear sizes for input, weight and output and the number of calls.
```
{(torch.Size([1, 16384, 512]), torch.Size([512, 512]), torch.Size([1, 16384, 512])): 4,
 (torch.Size([2, 96, 3840]), torch.Size([3840, 3840]), torch.Size([2, 96, 3840])): 8,
 (torch.Size([2, 96, 3840]), torch.Size([10240, 3840]), torch.Size([2, 96, 10240])): 4,
 (torch.Size([2, 96, 10240]), torch.Size([3840, 10240]), torch.Size([2, 96, 3840])): 2,
 (torch.Size([2, 256]), torch.Size([1024, 256]), torch.Size([2, 1024])): 1,
 (torch.Size([2, 256]), torch.Size([3840, 256]), torch.Size([2, 3840])): 1,
 (torch.Size([2, 256]), torch.Size([15360, 256]), torch.Size([2, 15360])): 32,
 (torch.Size([2, 1024]), torch.Size([256, 1024]), torch.Size([2, 256])): 1,
 (torch.Size([2, 4096, 3840]), torch.Size([3840, 3840]), torch.Size([2, 4096, 3840])): 8,
 (torch.Size([2, 4096, 3840]), torch.Size([10240, 3840]), torch.Size([2, 4096, 10240])): 4,
 (torch.Size([2, 4096, 10240]), torch.Size([3840, 10240]), torch.Size([2, 4096, 3840])): 2,
 (torch.Size([2, 4192, 3840]), torch.Size([64, 3840]), torch.Size([2, 4192, 64])): 1,
 (torch.Size([2, 4192, 3840]), torch.Size([3840, 3840]), torch.Size([2, 4192, 3840])): 120,
 (torch.Size([2, 4192, 3840]), torch.Size([10240, 3840]), torch.Size([2, 4192, 10240])): 60,
 (torch.Size([2, 4192, 10240]), torch.Size([3840, 10240]), torch.Size([2, 4192, 3840])): 30,
 (torch.Size([128, 2560]), torch.Size([3840, 2560]), torch.Size([128, 3840])): 1,
 (torch.Size([8192, 64]), torch.Size([3840, 64]), torch.Size([8192, 3840])): 1}
```
 ## Product

 - Should we use quantized version? stable-diffusion.cpp?
 - Run on what type of GPU? 
