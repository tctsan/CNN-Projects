<h1>Object Recognition</h1>

<h3>Task</h3>
<ul>
  <li>Task 1: Training with a Small Amount of Data</li>
  <li>Task 2: One-Shot / Few-Shot / Low-Shot Learning</li>
</ul>

</br>

<h2>Task 1 : Training with a Small Amount of Data</h2>
<ul>
  <li>Train a classifier to beat Microsoft Custom Vision AI baseline</li>
  <li>Use a limited version of Fashion MNIST dataset</li>
</ul>

<h3>Dataset</h3>

　<b>Fashion Mnist (Simplified)</b> : https://github.com/zalandoresearch/fashion-mnist</br>
 
<ul>
  <li>Training / Testing: <b>2K labeled</b> / 10K unlabeled images</li>
  <li>The images are split equally in <b>10 classes</b></li>
</ul>

![fashion-mnist-dataset](https://cdn-images-1.medium.com/max/800/1*PtQ2I-3RIFiCypor4u_Nbg.jpeg)

<h3>Method</h3>
　<b>Simple CNN model</b>

<img src="./image/fig_task1_architecture.png" alt="task1-method" width="40%" height="40%">

　<b>Parameters:</b></br>
　-- learning rate = 0.0001　　-- batch size = 128　　-- epoch = 300</br></br>

<h3>Learning Curve</h3>
　Training Loss & Training Accuracy</br>

<img src="./image/fig_task1_train_learning_curve.jpg" alt="task1-lr-curve" width="80%" height="80%">

<h3>Testing Results</h3>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>79.90%</td>
    </tr>
    <tr>
      <td>Our</td>
      <td>86.11%</td>
    </tr>
  </tbody>
</table>

</br>

<h2>Task 2 : One-Shot / Few-Shot / Low-Shot Learning</h2>
<ul>
  <li>Design a model to recognize a number of novel classes with insufficient number of training images</li>
</ul>

<h3>Dataset</h3>

　<b>Cifar-100 (Customized)</b> : https://www.cs.toronto.edu/~kriz/cifar.html</br>
 
<ul>
  <li><b>Training</b></li>
  <ul>
    <li><b>Base (80 classes)</b> : 500 images per class for training and 100 images per class for testing</li>
    <li>
      <b>Novel (20 classes)</b> : 500 images per class for training (need to <b>randomly pick few examples (1, 5, or 10)</b> during the training stage to simulate the few-shot setting)</li>
  </ul>
</ul>

<ul>
  <li><b>Testing</b></li>
  <ul>
    <li><b>2K unlabeled</b> images for <b>Novel</b> classes</li>
  </ul>
</ul>

![cifar100-dataset](https://cdn-images-1.medium.com/max/935/1*fTQtXyApxWPoW2vzSEk_Pw.png)

<h3>Method</h3>
　<b>Triplet Network + KNN</b></br>

<img src="./image/fig_task2_method.png" alt="task2-method" width="70%" height="70%">

　<b>Parameters:</b></br>
　-- learning rate = 0.0001　　-- batch size = 120　　-- epoch = 600</br></br>

<h3>Learning Curve</h3>
　Pre-training Loss & Training Loss</br></br>

<img src="./image/fig_task2_train_learning_curve.jpg" alt="task2-lr-curve" width="80%" height="80%">

<h3>Testing Results</h3>
<table>
  <thead>
    <tr>
      <th></th>
      <th>One-shot</th>
      <th>Five-shot</th>
      <th>Ten-shot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baseline</td>
      <td>20.00%</td>
      <td>46.50%</td>
      <td>52.30%</td>
    </tr>
    <tr>
      <td>Triplet Network + KNN</td>
      <td>30.40%</td>
      <td>52.55%</td>
      <td>57.20%</td>
    </tr>
  </tbody>
</table>

