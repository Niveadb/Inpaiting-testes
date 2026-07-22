'''
Código para inpainting usando OpenCV, para executar em alguma imagem
edite o caminho da imagem e da máscara nas linhas 19 e 20 e a pasta 
de saída na linha 21, e execute o código.

Implementação adaptada a partir do notebook:
Vardan Agarwal. "opencv_inpainting.ipynb".
https://colab.research.google.com/gist/vardanagarwal/094836d2876cdb045714424d6841ed23/opencv_inpainting.ipynb
Foram realizadas modificações na estrutura do código e na implementação
para atender aos objetivos deste projeto.
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, 'campo.jpeg')
mask_path = os.path.join(base_dir, 'campo_mask.jpg')
output_dir = os.path.join(base_dir, 'result_images_campo')

print(f'Carregando imagem: {img_path}')
print(f'Carregando máscara: {mask_path}')

img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f'Não foi possível abrir a imagem: {img_path}')
if mask is None:
    raise FileNotFoundError(f'Não foi possível abrir a máscara: {mask_path}')

img = cv2.resize(img, None, fx=0.25, fy=0.25)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
mask1 = cv2.bitwise_not(mask)
distort = cv2.bitwise_and(img, img, mask=mask1)

output1 = cv2.inpaint(distort, mask, 3, cv2.INPAINT_NS)
output2 = cv2.inpaint(distort, mask, 3, cv2.INPAINT_TELEA)

restored1 = img.copy()
restored2 = img.copy()
cv2.xphoto.inpaint(distort, mask1, restored1, cv2.xphoto.INPAINT_FSR_FAST)
cv2.xphoto.inpaint(distort, mask1, restored2, cv2.xphoto.INPAINT_FSR_BEST)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(distort, cv2.COLOR_BGR2RGB)
dst1 = cv2.cvtColor(output1, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(output2, cv2.COLOR_BGR2RGB)
dst3 = cv2.cvtColor(restored1, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(restored2, cv2.COLOR_BGR2RGB)

os.makedirs(output_dir, exist_ok=True)

def save_image(path, image, ext='.jpg'):
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f'Falha ao codificar a imagem: {path}')
    with open(path, 'wb') as f:
        f.write(buf.tobytes())

save_image(os.path.join(output_dir, 'original.jpg'), img, '.jpg')
save_image(os.path.join(output_dir, 'distorted.jpg'), distort, '.jpg')
save_image(os.path.join(output_dir, 'inpaint_ns.jpg'), output1, '.jpg')
save_image(os.path.join(output_dir, 'inpaint_telea.jpg'), output2, '.jpg')
save_image(os.path.join(output_dir, 'fsr_fast.jpg'), restored1, '.jpg')
save_image(os.path.join(output_dir, 'fsr_best.jpg'), restored2, '.jpg')
print(f'Imagens individuais salvas em: {output_dir}')

plt.figure(figsize=(15, 20))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(dst)
plt.title('Distorted')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dst1)
plt.title('Inpaint NS')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(dst2)
plt.title('Inpaint TELEA')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(dst3)
plt.title('FSR FAST')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(dst4)
plt.title('FSR BEST')
plt.axis('off')

plt.tight_layout()
plt.show()