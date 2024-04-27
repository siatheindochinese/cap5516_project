psnr_values = []
for d in document_content.values():
    if not isnan(d.get('psnr', nan)):
        psnr_values.append(d['psnr'])

ssim_values = []
for d in document_content.values():
    if not isnan(d.get('ssim', nan)):
        ssim_values.append(d['ssim'])


import numpy as np

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average PSNR: {avg_psnr:.3f}")
print(f"Average SSIM: {avg_ssim:.3f}")