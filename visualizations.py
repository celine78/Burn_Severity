plt.subplot(1,2,1)
plt.imshow(1-masks[0],cmap='gray')
plt.subplot(1,2,2)
plt.hist(1-masks[0])

plt.subplot(1,2,1)
plt.imshow(1-masks_normalized[0].squeeze(), cmap='gray')
plt.subplot(1,2,2)
plt.hist(1-masks_normalized[0].squeeze())

hist_01 = np.histogram(masks[0].squeeze(), density=True)
hist_02 = np.histogram(masks_normalized[0].squeeze(), density=True)
fig, ax = plt.subplots(2,1, figsize=(5,5))
#ax[0].hist(hist_01, facecolor='#75be25')
#ax[1].hist(hist_02, facecolor='#3b4cc5') #165a72 #3b4cc5 #be2538 #75be25
ax[0].hist(hist_01)
ax[1].hist(hist_02)

image_mask = 1-masks_normalized[1].squeeze().numpy()
t = filters.threshold_otsu(image_mask,np.histogram(image_mask)[1])
t2 = filters.threshold_minimum(image_mask,np.histogram(image_mask)[1])
#out = (1-masks_normalized[1]>=t).int()
binary_mask = image_mask >= t
binary_mask2 = image_mask >= t2
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax[0,0].imshow(image_mask, cmap='gray')
#ax[0,1].imshow(out.squeeze(), cmap='gray')
ax[1,0].imshow(binary_mask, cmap='gray')
ax[1,1].imshow(binary_mask2, cmap='gray')
#print(binary_mask)
#print(out)

from skimage.filters import try_all_threshold
image_mask = 1-masks_normalized[0].squeeze().numpy()
fig, ax = try_all_threshold(image_mask, figsize=(10, 20), verbose=False)
plt.show()

image_mask = 1-masks_normalized[0].squeeze().numpy()
thresholds = filters.threshold_multiotsu(image_mask,4,np.histogram(image_mask)[1])
regions = np.digitize(image_mask, bins=thresholds)
#regions[regions == 0] = 1
#regions[regions == 1] = 0
#regions[regions == 2] = 0
#regions[regions == 3] = 0
print(thresholds)
print(np.unique(regions))
print(np.unique(image_mask))
print(type(regions))
print(regions.shape)
print(regions)
fig, ax = plt.subplots(1,2,figsize=(20,20))
ax[0].imshow(image_mask, cmap='gray')
ax[1].imshow(regions, cmap='gray')



im = images[0]
ma = masks[0]

im_vflip = randomVflip(im,1)
im_hflip = randomHflip(im,1)
im_rotation = randomRotation(im,1,40)
im_shear = randomShear(im,1,40)

mask_vflip = randomVflip(ma,1)
mask_hflip = randomHflip(ma,1)
mask_rotation = randomRotation(ma,1,40)
mask_shear = randomShear(ma,1,40)

fig, ax = plt.subplots(2,5, figsize=(20,10))
ax[0][0].imshow(images[0][:,:,4])
ax[0][1].imshow(im_vflip[:,:,4])
ax[0][2].imshow(im_hflip[:,:,4])
ax[0][3].imshow(im_rotation[:,:,4])
ax[0][4].imshow(im_shear[:,:,4])

ax[1][0].imshow(masks[0].squeeze(), cmap='gray')
ax[1][1].imshow(mask_vflip.squeeze(), cmap='gray')
ax[1][2].imshow(mask_hflip.squeeze(), cmap='gray')
ax[1][3].imshow(mask_rotation.squeeze(), cmap='gray')
ax[1][4].imshow(mask_shear.squeeze(), cmap='gray')

