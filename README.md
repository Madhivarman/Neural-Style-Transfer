# Neural-Style-Transfer
Python Flask application to apply NST to Normal Images

## Run the Program ##

To transfer the style to the Content Image pass this command

``` python model.py --baseimg images/content_image.jpg --styleimg images/style_image.jpg --poolingChoice max  --max_iterations 1500 --print_iterations 100 --result_name output_1 ```

To know the descriptions of the argument run this command

``` python model.py --help ```

### MODEL ARCHITECTURE ###

```
Constructing Layers......
--conv1_1 | shape=(1, 512, 288, 64) | weights_shape=(3, 3, 3, 64)
--relu_1 | shape=(1, 512, 288, 64) | Bias_shape=(64,)
--conv1_2 | shape=(1, 512, 288, 64) | weights_shape=(3, 3, 64, 64)
--relu1_2 | shape=(1, 512, 288, 64) | Bias_shape=(64,)
--pool1 | Shape=(1, 512, 288, 64)
--conv2_1 | shape=(1, 256, 144, 128) | weights_shape=(3, 3, 64, 128)
--conv2_1 | shape=(1, 256, 144, 128) | weights_shape=(3, 3, 64, 128)
--relu2_1 | shape=(1, 256, 144, 128) | Bias_shape=(128,)
--conv2_2 | shape=(1, 256, 144, 128) | weights_shape=(3, 3, 128, 128)
--relu2_2 | shape=(1, 256, 144, 128) | Bias_shape=(128,)
--pool2 | Shape=(1, 256, 144, 128)
--conv3_1 | shape=(1, 128, 72, 256) | weights_shape=(3, 3, 128, 256)
--relu3_1 | shape=(1, 128, 72, 256) | Bias_shape=(256,)
--conv3_2 | shape=(1, 128, 72, 256) | weights_shape=(3, 3, 256, 256)
--relu3_2 | shape=(1, 128, 72, 256) | Bias_shape=(256,)
--conv3_3 | shape=(1, 128, 72, 256) | weights_shape=(3, 3, 256, 256)
--relu3_3 | shape=(1, 128, 72, 256) | Bias_shape=(256,)
--conv3_4 | shape=(1, 128, 72, 256) | weights_shape=(3, 3, 256, 256)
--relu3_4 | shape=(1, 128, 72, 256) | Bias_shape=(256,)
--pool3 | Shape=(1, 128, 72, 256)
--conv4_1 | shape=(1, 64, 36, 512) | weights_shape=(3, 3, 256, 512)
--relu4_1 | shape=(1, 64, 36, 512) | Bias_shape=(512,)
--conv4_2 | shape=(1, 64, 36, 512) | weights_shape=(3, 3, 512, 512)
--relu4_2 | shape=(1, 64, 36, 512) | Bias_shape=(512,)
--conv4_3 | shape=(1, 64, 36, 512) | weights_shape=(3, 3, 512, 512)
--relu4_3 | shape=(1, 64, 36, 512) | Bias_shape=(512,)
--conv4_4 | shape=(1, 64, 36, 512) | weights_shape=(3, 3, 512, 512)
--relu4_4 | shape=(1, 64, 36, 512) | Bias_shape=(512,)
--pool4 | Shape=(1, 64, 36, 512)
--conv5_1 | shape=(1, 32, 18, 512) | weights_shape=(3, 3, 512, 512)
--relu5_1 | shape=(1, 32, 18, 512) | Bias_shape=(512,)
--conv5_2 | shape=(1, 32, 18, 512) | weights_shape=(3, 3, 512, 512)
--relu5_2 | shape=(1, 32, 18, 512) | Bias_shape=(512,)
--conv5_3 | shape=(1, 32, 18, 512) | weights_shape=(3, 3, 512, 512)
--relu5_3 | shape=(1, 32, 18, 512) | Bias_shape=(512,)
--conv5_4 | shape=(1, 32, 18, 512) | weights_shape=(3, 3, 512, 512)
--relu5_4 | shape=(1, 32, 18, 512) | Bias_shape=(512,)
--pool5 | Shape=(1, 32, 18, 512)
```

## NOTE ##
So far only core work is finished! UserInterface, other assignments will updated in frequent time events.

# Citations

The Model Source code is Heavily based on this refered citation.

```
@author{
  'Name': 'Madhivarman',
  'ProjectName': 'NeuralStyleTransfer',
  'Description': 'A Simple Application transfer styles from Painting Image to Content Image'
  'ReferedRepository': 'https://github.com/cysmith/neural-style-tf'
} 
```
