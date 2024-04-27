import json
import numpy as np
from numpy import isnan, nan

document_content = json.load(open('240.json'))

psnr_values = []
for d in document_content.values():
    if not isnan(d.get('psnr', nan)):
        psnr_values.append(d['psnr'])

ssim_values = []
for d in document_content.values():
    if not isnan(d.get('ssim', nan)):
        ssim_values.append(d['ssim'])

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print('x1 reconstruction:')
print(f"Average PSNR: {avg_psnr:.3f}")
print(f"Average SSIM: {avg_ssim:.3f}")

document_content = json.load(open('120.json'))

psnr_values = []
for d in document_content.values():
    if not isnan(d.get('psnr', nan)):
        psnr_values.append(d['psnr'])

ssim_values = []
for d in document_content.values():
    if not isnan(d.get('ssim', nan)):
        ssim_values.append(d['ssim'])

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print('x2 reconstruction:')
print(f"Average PSNR: {avg_psnr:.3f}")
print(f"Average SSIM: {avg_ssim:.3f}")

document_content = json.load(open('60.json'))

psnr_values = []
for d in document_content.values():
    if not isnan(d.get('psnr', nan)):
        psnr_values.append(d['psnr'])

ssim_values = []
for d in document_content.values():
    if not isnan(d.get('ssim', nan)):
        ssim_values.append(d['ssim'])

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print('x4 reconstruction:')
print(f"Average PSNR: {avg_psnr:.3f}")
print(f"Average SSIM: {avg_ssim:.3f}")
