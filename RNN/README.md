<h1>Action Recognition</h1>

<h3>Goal</h3>
<ul>
  <li>Ability to extract state-of-the-art deep CNN features</li>
  <li>Implement Recurrent Neural Networks (RNN) for action recognition</li>
  <li>Extend RNN models for solving sequence-to-sequence problems</li>
</ul>

<h3>Dataset</h3>

![dataset](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/dataset.png)

</br>
<ul>
  <li>Total 37 full-length videos (each 5-20 mins in 24 fps)</li>
  <li>Split into 4151 trimmed videos (each 5-20 secs in 24 fps)</li>
  <li>Extract frame from video (fps=2)</li>
  <li>11 action classes</li>
</ul>

</br>

<h2>Task 1 : Data Preprocessing</h2>
<ul>
  <li>Extract state-of-the-art CNN features for action recognition</li>
  <li>Use <b>VGG16 pre-trained model</b></li>
</ul>

<h3>Model</h3>

![CNN-model](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/CNN-model.png)

</br>
　<b>Parameters:</b></br>
　-- learning rate = 0.0001　　-- batch size = 64　　-- epoch = 30</br></br>

<h3>Learning Curve</h3>
　Training Loss, Training Accuracy & Validation Accuracy

![CNN-lr_curve](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/CNN-learning-curve.jpg)

</br>
　Training Accuracy = 0.953125
　Validation Accuracy = 0.470019 (No significant improvement after accu=0.4)</br></br>


<h2>Task 2 : Trimmed Action Recognition</h2>
<ul>
  <li>Training a RNN model with sequences of CNN features and labels</li>
</ul>

<h3>Model</h3>

![RNN-model](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/RNN-model.png)

</br>
　<b>Parameters:</b></br>
　-- learning rate = 0.0001　　-- batch size = 32　　-- epoch = 120</br></br>

<h3>Learning Curve</h3>
　Training Loss & Validation Accuracy

![RNN-lr_curve](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/RNN-learning-curve.jpg)

</br>
　Validation Accuracy = 0.458414</br></br>

<h3>Feature Visualization</h3>
　Visualize CNN-based video features and RNN-based video features to 2D space (with t-SNE)

![feature-visual](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/feature-visualization.png)

</br>

<h2>Task 3 : Temporal Action Segmentation</h2>
<ul>
  <li>Extend a RNN model for sequence-to-sequence prediction</li>
</ul>

<h3>Model</h3>

![RNN-seq2seq-model](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/RNN-seq2seq-model.png)

</br>
　<b>Parameters:</b></br>
　-- learning rate = 0.0001　　-- batch size = 64　　-- epoch = 120</br></br>

<h3>Learning Curve</h3>
　Training Loss, Training Accuracy & Validation Accuracy

![RNN-seq2seq-lr_curve](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/RNN-seq2seq-learning-curve.jpg)

</br>
　Validation Accuracy = 0.453187 (No significant improvement after accu=0.4)</br></br>

<h3>Feature Visualization</h3>
　Visualize the prediction result in comparison with the ground-truth

![RNN-seq2seq-visual](https://github.com/tctsan/CNN-Projects/blob/master/RNN/image/RNN-seq2seq-visualization.png)

</br>
　Label 0 occupies almost all the predictions (need to improve...)
