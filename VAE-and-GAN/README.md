
<h1>Image Generation and Feature Disentanglement</h1>

<h3>Goal</h3>
<ul>
  <li>Ability to handle large-scale human face data with deep neural network</li>
  <li>Learn and implement well-known image generation models</li>
  <li>Gain experience of adversarial training</li>
  <li>Supervised/unsupervised feature disentanglement</li>
</ul>

<h3>Dataset</h3>
<ul>
  <li>CelebA</b>: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html</li>
</ul>
</br>

<h2>VAE (Variational Auto-Encoder)</h2>

<h3>Model</h3>

![VAE-model](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/VAE-model.png)

</br>
　<b>Parameters:</b></br>
　-- lambda_KL = 0.001　-- learning rate = 0.001　　-- batch size = 128　　-- epoch = 50</br>

<h3>Training</h3>

![VAE-training](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/VAE-training.gif)

<h3>Learning Curve</h3>
　MSE(Reconstruction loss) & KLD (KL divergence)</br>

![VAE-lr_curve](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/VAE-learning-curve.jpg)

<h3>Reconstruction Results</h3>
　Testing Images and the Reconstructed Results</br>

![VAE-results](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/VAE-results.jpg)

</br>
　Testing MSE = 0.261635</br></br>


<h2>GAN (Generative Adversarial Network)</h2>

<h3>Model</h3>

![GAN-model](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/GAN-model.png)

</br>
　<b>Parameters:</b></br>
　-- learning rate of G = 0.001　　-- learning rate of D = 0.003　　-- batch size = 32　　-- epoch = 50</br>

<h3>Training</h3>

![GAN-training](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/GAN-training.gif)

<h3>Learning Curve</h3>
　Training Loss and Accuracy

![GAN-lr_curve](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/GAN-learning-curve.jpg)

<h3>Generated Results</h3>
　Random Generated Images</br></br>

![GAN-results](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/GAN-results.jpg)

</br>

<h2>ACGAN (Auxiliary Classifier Generative Adversarial Network)</h2>

<h3>Model</h3>

![ACGAN-model](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/ACGAN-model.png)

</br>
　<b>Parameters:</b></br>
　-- learning rate of G = 0.001　　-- learning rate of D = 0.003　　-- batch size = 32　　-- epoch = 34</br>

<h3>Training</h3>

![ACGAN-training](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/ACGAN-training.gif)

<h3>Learning Curve</h3>
　Training Loss of Attribute Classification and Accuracy of Discriminator 

![ACGAN-lr_curve](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/ACGAN-learning-curve.jpg)

<h3>Generated Results</h3>
　Random Generated Images (Attribute: Smiling and No Smiling) </br></br>

![ACGAN-results](https://github.com/tctsan/CNN-Projects/blob/master/VAE-and-GAN/image/ACGAN-results.jpg)

</br>

<h2>References</h2>

<ul>
  <li><a href="https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners?imm_mid=0f3eba&cmp=em-data-na-na-newsltr_20170628" rel="nofollow">
    Generative Adversarial Networks for beginners</a>
  </li>
  <li><a href="https://gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html" rel="nofollow">
    Deep Convolutional Generative Adversarial Networks</a>
  </li>
  <li><a href="https://github.com/jonbruner/generative-adversarial-networks" rel="nofollow">
      jonbruner/generative-adversarial-networks</a>
  </li>
</ul>
