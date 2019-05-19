<h1>Handwritten Digit Recognition</h1>

<h3>Goal</h3>
<ul>
  <li>Build a CNN model and train it on MNIST dataset</li>
  <li>Visualize the filter in different layers</li>
  <li>Visualize the low-level and high-level features</li>
</ul>

<h3>Dataset</h3>
<ul>
  <li><b>MNIST</b> : http://yann.lecun.com/exdb/mnist/</li>
</ul>

<h3>Model</h3>

![CNN-model](./image/cnn_architecture.png)

</br>
　<b>Parameters:</b></br>
　-- learning rate = 0.0001　　-- batch size = 256　　-- epoch = 100</br>

<h3>Learning Curve</h3>
　Training Accuracy & Loss</br></br>
 
![CNN-lr-curve](./image/learning curves.png)

　Training   Accuracy = 1.0</br>
　Validation Accuracy = 0.9928</br></br>

<h3>Filter Visualization</h3>
　The first convolutional layers (total 32 filters)</br>

![filter-visual-first](./image/filter_visualization_conv_first.png)

　The last convolutional layers (total 128 filters)</br>

![filter-visual-last](./image/filter_visualization_conv_last.png)

<h3>Feature Visualization</h3>
　The Low-Level Features</br>
 
![feature-visual-low](./image/features_visualization_low-level.jpg)

　The High-Level Features</br>
 
![feature-visual-high](./image/features_visualization_high-level.jpg)

</br>

<h2>References</h2>

<ul>
  <li><a href="https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html" rel="nofollow">
    How Convolutional Neural Networks See the World</a>
  </li>
</ul>
