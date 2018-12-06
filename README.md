# Readme

## How to run

### Training

The following command will train your neural network from scratch on 10 epochs. On my computer it takes around 40s each epoch :
```python
glowing-spoon --train=True --save_nn="model.pth" --epochs=10
```

### Testing

In order to test, you need to specify the testing flag and some images to test. For example :
```python
glowing-spoon --train=False --test=True
    --x_test="/new/path/folder/test/modified/"
    --y_test="/new/path/folder/test/original/"
    --load_nn="model.pth"
    --print_example=True
```

## Remarks

 * By running : 
   ```python
    glowing-spoon --help
    ```
    you will recieve a lot of indication and the default values for each parameters.

 * The measurement is the structural similarity from scikit-learn. The mean value is about 0.80 over the whole images.

 * I didn't have enough time to performe cross-validation neather k-fold validation and to port the code on GPU.

 * This isn't a perfect tool, there is a lot of improvement that can be done. It's not very stable and this code is mostly a proof of concept of your "research question".