*Lablr* is label/tag recommendation tool for flickr images allowing the users to use the most relevant labels for their images to maximize the viewship of their photos. Lablr suggests the most relevant label from the [all time most popular tags used on flickr](https://www.flickr.com/photos/tags).

Web app
=======
Lablr is freely available as a [web app](http://3.17.244.100).

Author information
==================
My name is [Aras Kayvanrad](https://www.linkedin.com/in/kayvanrad/) and I developed Lablr during my fellowship at the [Insight Data Science program](https://www.insightdatascience.com/).

Implementation details
======================
Lablr uses a [variational autoencoder (VAE)](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_.28VAE.29) model for label suggestion. A VAE model is trained on a training set of flickr images. Once the VAE model is trained, the encoder block is used to construct the latent space. Different labels have distinct distributions in the latent space.

![Latent space distributions](https://github.com/kayvanrad/lablr/blob/master/images/Latent%20space%20distributions.png)

For any new image, the proximity of the image to each distribution is measured using the [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance). The closest distributions are the most relevant labels for that image.

In addition to suggesting the most relevant labels, Lablr provides a visualization of the latent space showing where the new image falls relative to the most relevant labels. This visualization is a 2D PCA plot of the latent space. The centroid and dispersion of each disribution are visualized using circles centred on the centroid of the distribution with radii proportional to the dispersion, where the dispersion is measured as the determinant of the corresponding covariance matrix.

![latent space visualization](https://github.com/kayvanrad/lablr/blob/master/images/latent_space20190308-022520.png]
