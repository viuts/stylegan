import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

from training import misc

def main():
    tflib.init_tf()
    network_pkl, resume_kimg = misc.locate_latest_pkl()
    _G, _D, Gs = pickle.load(open(network_pkl, "rb"))
    Gs.print_layers()

    for i in range(0,1000):
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
        os.makedirs(os.path.join(config.result_dir, 'example'), exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example','example-'+str(i)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()