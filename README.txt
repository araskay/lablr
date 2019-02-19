Lablr: Analyze and visualize the distribution of popular image labels on Flickr, and suggest labels for new images based on the distribution

To get started:
1- download images from flickr using flickr_download.py
2- resize downloaded images to a common size, and create a csv file with the path/file_name of all images and their label, using flickr_resize.py
3- train a VAE model using flickr_vae_gen.py (or flickr_vae.py if you feel confident about your memory or using HPC clusters)
4- run the web app (webapp.py)
