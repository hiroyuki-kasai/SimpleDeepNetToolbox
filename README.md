# SimpleDeepNetToolbox : Simple Deep Net Toolbox in MATLAB
----------

Authors: [Hiroyuki Kasai](http://kasai.kasailab.com/)

Last page update: October 09, 2018

Latest library version: 1.0.0 (see Release notes for more info)

<br />

Introduction
----------
The SimpleDeepNetToolbox is a **pure-MATLAB** and simple toolbox for deep learning.

There are much better other toolboxes available for deep learning, e.g. Theano, torch or tensorflow. 
I would definitely recomment you to use one of such tools for your problems at hand. 
The main purpose of this toolbox is to allows you, especially "MATLAB-lover" researches, to undedrstand deep learning techniques using "non-black-box" simple implementations. 

<br />



## <a name=""> List of the network available in SimpleDeepNetToolbox </a>


- **Feedforward Backpropagation Neural Networks** 
- **Convolutional Neural Networks**

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


Second to do: download datasets and external libraries
----------------------------
Run `download` for downloading datasets and external libraries.
```Matlab
%% Run the downloading script
download;
```

- If your computer is behind a proxy server, please configure your Matlab setting. See [this](http://jp.mathworks.com/help/matlab/import_export/proxy.html?lang=en).


<br />


Simplest usage example: 4 steps!
----------------------------

Just execute `demo_multilayer_neuralnet` for the simplest demonstration of this package. This is a forward backward neural network.

```Matlab
%% Execute the demonstration script
demo_multilayer_neuralnet;
```

The "**demo_multilayer_neuralnet.m**" file contains below.
```Matlab
% set parameters
max_epoch = 30;
batch_size = 100;
opt_alg = 'SGD';
learning_rate = 0.1;


% load dateaset
dataset_dir = './datasets/';
dataset_name = 'mnist';
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset(dataset_name, dataset_dir, 5000, 1000, false);


% set network
network = multilayer_neural_net(dimension, [100 80], class_num, 'relu', 'relu', 0.01, 0, 0, 0, opt_alg, 0.1);


% set trainer
trainer = trainer(network, x_train, t_train, x_test, t_test, max_epoch, batch_size, 0, 1);


% train
tic             
[info] = trainer.train(); 
elapsedTime = toc;
fprintf('elapsed time = %5.2f [sec]\n', elapsedTime);
```

<br />
Let take a closer look at the code above bit by bit. The procedure has only **4 steps**!

**Step 1: Load dataset**

First, we load a dataset including train set and test set using a data loader function `load_dataset()`. 
The output include train set and test set, and related other data.

```Matlab    
dataset_dir = './datasets/';
dataset_name = 'mnist';
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset(dataset_name, dataset_dir, 5000, 1000, false);
```

**Step 2: Set network**

The problem to be solved should be defined properly from the [supported problems](#supp_pro). `logistic_regression()` provides the comprehensive 
functions for a logistic regression problem. This returns the cost value by `cost(w)`, the gradient by `grad(w)` and the hessian by `hess(w)` when given `w`. 
These are essential for any gradient descent algorithms.
```Matlab
network = multilayer_neural_net(dimension, [100 80], class_num, 'relu', 'relu', 0.01, 0, 0, 0, opt_alg, 0.1);
```

**Step 3: Perform solver**

Now, you can perform optimization solvers, i.e., SGD and SVRG, calling [solver functions](#supp_solver), i.e., `sgd()` function and `svrg()` function after setting some optimization options. 
```Matlab
options.w_init = data.w_init;
options.step_init = 0.01;  
[w_sgd, info_sgd] = sgd(problem, options);  
[w_svrg, info_svrg] = svrg(problem, options);
```
They return the final solutions of `w` and the statistics information that include the histories of epoch numbers, cost values, norms of gradient, the number of gradient evaluations and so on.

**Step 4: Show result**

Finally, `display_graph()` provides output results of decreasing behavior of the cost values in terms of the number of gradient evaluations. 
Note that each algorithm needs different number of evaluations of samples in each epoch. Therefore, it is common to use this number to evaluate stochastic optimization algorithms instead of the number of iterations.
```Matlab
display_graph('grad_calc_count','cost', {'SGD', 'SVRG'}, {w_sgd, w_svrg}, {info_sgd, info_svrg});
```

That's it!

<br />

More plots
----------------------------


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

Release Notes
--------------
* Version 1.0.0 (Oct. 08, 2018)
    - Initial version.




