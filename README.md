<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Self-Driving-Car-Lane-Detection-using-UNet-Model</h1>

<p>This project implements lane detection using both classical image processing techniques and a deep learning approach with a U-Net model. The pipeline processes images and videos to detect lanes on roads.</p>

<h2>Project Structure</h2>
<ul>
    <li><strong>Classical_utils.py</strong>: Contains utility functions for image processing (e.g., grayscale conversion, Gaussian blur, Canny edge detection, Hough transform).</li>
    <li><strong>Classical.py</strong>: Implements a classical lane detection pipeline using the functions in <code>classical_utils.py</code>.</li>
    <li><strong>Draw_lanes.py</strong>: Applies the trained U-Net model to detect lanes in a video file.</li>
    <li><strong>predict.py</strong>: Uses a trained U-Net model to predict lanes on a given image and calculates Intersection over Union (IoU).</li>
    <li><strong>training.py</strong>: Trains the U-Net model on a dataset of images and their corresponding lane masks.</li>
    <li><strong>unet.py</strong>: Defines the U-Net model architecture.</li>
    <li><strong>Visualize.py</strong>: Visualizes training and test images for inspection.</li>
</ul>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/Bhargavi-Joshi/Self-Driving-Car-Lane-Detection-using-UNet-Model.git</code></pre>
    </li>
    <li>Install the required packages:
        <pre><code>pip install -r Requirements.txt</code></pre>
    </li>
</ol>

<h2>Usage</h2>

<h3>Classical Lane Detection</h3>
<pre><code>python Classical.py &lt;image_index&gt;</code></pre>

<h3>U-Net Model Training</h3>
<pre><code>python training.py</code></pre>

<h3>Predict Lanes</h3>
<pre><code>python predict.py &lt;image_index&gt;</code></pre>

<h3>Draw Lanes on Video</h3>
<pre><code>python Draw_lanes.py &lt;path_to_video&gt;</code></pre>

<h3>Visualize Dataset</h3>
<pre><code>python Visualize.py &lt;number_of_images&gt; &lt;start_index&gt;</code></pre>

<h2>Dataset</h2>
<p>The training and labelled dataset used for the lane detection models can be downloaded from the following link:</p>
<p><a href="https://1024terabox.com/s/1yl4k5BYNWXlRUpoFvMgkZA" target="_blank">Download Dataset</a></p>

<h2>Authors</h2>
<p>Developed by:</p>
<ul>
    <li><strong>Pranjal Gautam</strong> - <a href="pranjalgautam1103@gmail.com">pranjalgautam1103@gmail.com</a></li>
    <li><strong>Bhargavi Joshi</strong> - <a href="mailto:bhargavijoshi86@gmail.comm">bhargavijoshi86@gmail.com</a></li>
</ul>
</body>
</html>
