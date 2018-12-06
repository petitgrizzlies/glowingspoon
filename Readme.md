# Readme

## How to run

### Requierement

You just need to get `pipenv` installed. It can be download with `pip install pipenv`.
Then just :

```bash
cd into/dir/where is core.py/
pipenv shell
# just run the following commands to train, test...
```

### Training

The following command will train your neural network from scratch on 10 epochs. On my computer it takes around 40s each epoch :
```python
python core.py --train=True --save_nn="model.pth" --epochs=10
```

### Testing

In order to test, you need to specify the testing flag and some images to test. For example :
```python
python core.py --train=False --test=True
    --x_test="/new/path/folder/test/modified/"
    --y_test="/new/path/folder/test/original/"
    --load_nn="model.pth"
    --print_example=True
```

## Remarks

 * By running : 
   ```python
    python core.py --help
    ```
    you will recieve a lot of indication and the default values for each parameters.

 * The measurement is the structural similarity from scikit-learn. The mean value is about 0.80 over the whole images.

 * I didn't have enough time to performe cross-validation neather k-fold validation and to port the code on GPU.

 * This isn't a perfect tool, there is a lot of improvement that can be done. It's not very stable and this code is mostly a proof of concept of your "research question".