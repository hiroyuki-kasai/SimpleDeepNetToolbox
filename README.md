# SimpleDeepNetToolbox : Simple Deep Net Toolbox in MATLAB
----------

Authors: [Hiroyuki Kasai](http://kasai.kasailab.com/)

Last page update: November, 09, 2018

Latest library version: 1.0.2 (see Release notes for more info.)

<br />

Introduction
----------
The SimpleDeepNetToolbox is a **pure-MATLAB** and simple toolbox for deep learning. This toolbox was originally ported from [python library](https://github.com/oreilly-japan/deep-learning-from-scratch). However, major modification have been made for MATLAB implementation and its efficient implementation.

There are much better other toolboxes available for deep learning, e.g. Theano, torch or tensorflow. 
I would definitely recommend you to use one of such tools for your problems at hand. 
The main purpose of this toolbox is to allows you, especially "MATLAB-lover" researchers, to undedrstand deep learning techniques using "non-black-box" simple implementations. 

<br />



## <a name=""> List of available network architectures</a>


- **Feedforward Backpropagation Neural Networks** 
- **Convolutional Neural Networks**


## <a name=""> List of available layers</a>


- **Affine layer** 
- **Conolution layer**
- **Pooling layer**
- **Dropout layer**
- **Batch normalization layer** (Under construction)
- **ReLu (Rectified Linear Unit) layer**
- **Sigmoid layer**
- **Softmax layer**

## <a name=""> List of available optimization solvers</a>


- **Vanila SGD** 
- **AdaGrad**
- **Momentum SGD**


<br />

Folders and files
---------
<pre>
./                      - Top directory.
./README.md             - This readme file.
./run_me_first.m        - The scipt that you need to run first.
./demo.m                - Demonstration script to check and understand this package easily. 
./download.m            - Script to download datasets.
|networks/              - Contains various network classes.
|layers/               	- Contains various layer classes.
|optimizer/             - Contains optimization solvers.
|test_samples/          - Contains test samples.
|datasets/          	- Contains dasetsets (to be downloaded).
</pre>                       


<br />


First to do: configure path
----------------------------
Run `run_me_first` for path configurations.
```Matlab
%% First run the setup script
run_me_first;
```


Second to do: download datasets
----------------------------
Run `download` for downloading datasets.
```Matlab
%% Run the downloading script
download;
```

- If your computer is behind a proxy server, please configure your Matlab setting. See [this](http://jp.mathworks.com/help/matlab/import_export/proxy.html?lang=en).


<br />


Simplest usage example: 5 steps!
----------------------------

Just execute `demo_two_layer_neuralnet` for the simplest demonstration of this package. This is a forward backward neural network.

```Matlab
%% load dateaset
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset('mnist', './datasets/',  inf, inf, false);

%% set network
network = two_layer_net(x_train, t_train, x_test, t_test, 784, 50, 10, []);

%% set trainer
trainer = nn_trainer(network);


%% train
info = trainer.train(); 

%% plot
display_graph('epoch', 'cost', {'Tow layer net'}, {}, {info});    

train_info = info;
test_info = info;
train_info.accuracy = info.train_acc;
test_info.accuracy = info.test_acc;
display_graph('epoch', 'accuracy', {'Train', 'Test'}, {}, {train_info, test_info});   
```

<br />
<br />

Let's take a closer look at the code above bit by bit. The procedure has only **5 steps**!

**Step 1: Load dataset**

First, we load a dataset including train set and test set using a data loader function `load_dataset()`. 
The output include train set and test set, and related other data.

```Matlab    
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset('mnist', './datasets/',  inf, inf, false);
```

**Step 2: Set network**

The next step defines the network architecture. This example uses a two layer neural network with the input size 784, the hidden layer size 50, and the output layer size 10. Datasets are also delivered to this class.

```Matlab
%% set network
network = two_layer_net(x_train, t_train, x_test, t_test, 784, 50, 10, []);
```

**Step 3: Set trainer**

You also set the network to be used. Some options for training could be configured using the second argument, which is not used in this example, though. 

```Matlab
%% set trainer
trainer = nn_trainer(network);
```

**Step 4: Perform trainer**

Now, you start to train the network.  

```Matlab
%% train
info = trainer.train(); 
```
It returns the statistics information that include the histories of epoch numbers, cost values, train and test accuracies, and so on.

**Step 5: Show result**

Finally, `display_graph()` provides output results of decreasing behavior of the cost values in terms of the number of epoch. The accuracy results for the train and the test are also shown. 

```Matlab
% plot
display_graph('epoch', 'cost', {'Tow layer net'}, {}, {info});    

train_info = info;
test_info = info;
train_info.accuracy = info.train_acc;
test_info.accuracy = info.test_acc;
display_graph('epoch', 'accuracy', {'Train', 'Test'}, {}, {train_info, test_info}); 
```

That's it!

<br />

More plots
----------------------------

TBA.

<br />

License
-------
- The SimpleDeepNetToolbox is **free** and **open** source.
- The code provided iin SGDLibrary should only be used for **academic/research purposes**.
- This toolbox was originally ported from [python library](https://github.com/oreilly-japan/deep-learning-from-scratch). 


<br />

Notes
-------
- As always, parameters such as the learning rate should be configured properly in each dataset and network. 

<br />

Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.kasailab.com/) (email: kasai **at** is **dot** uec **dot** ac **dot** jp)

<br />

Release notes
--------------
* Version 1.0.2 (Nov. 09, 2018)
    - Some class structures are re-configured.
* Version 1.0.1 (Nov. 07, 2018)
    - Some class structures are re-configured.
* Version 1.0.0 (Oct. 08, 2018)
    - Initial version.




