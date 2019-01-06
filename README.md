# Neural-Style-Transfer
Python Flask application to apply NST to Normal Images

## Run the Program ##

To transfer the style to the Content Image pass this command

> python model.py --baseimg images/content_image.jpg --styleimg images/styleimg.jpg --poolingChoice max  --max_iterations 1500 --print_iterations 100 --result_name output_1

To know the descriptions of the argument run this command
> python model.py --help

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
