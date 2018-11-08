clc;
clear;
close all;

rng('default');

max_epoch = 30;
verbose = 1;
w_decay_lambda = 0.01; % l2-norm regularization. 0.1?? (0 ... 1)
%opt_alg = 'SGD';
%opt_alg = 'Momuentum';
opt_alg = 'AdaGrad';


if 0
    total_train_size = 60000;
    total_test_size = 10000;
    batch_size = 100;
    learning_rate = 0.1;
elseif 1
    total_train_size = 5000;
    total_test_size = 1000;
    batch_size = 100;
    learning_rate = 0.1;    
else
    total_train_size = 500;
    total_test_size = 500;
    batch_size = 10; 
    learning_rate = 0.1;
end


% load dateaset
dataset_dir = '../datasets/';
%dataset_name = 'mnist';
%dataset_name = 'orl_face';
%dataset_name = 'usps';
%dataset_name = 'coil100';
dataset_name = 'coil20';
%dataset_name = 'cifar-100';
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, height, width] = ...
    load_dataset(dataset_name, dataset_dir, total_train_size, total_test_size, true);
     

% set network
input_dim = [1, height, width];
conv_param = [];
conv_param.filter_num = 30;
conv_param.filter_size = 5;
conv_param.pad = 0;
conv_param.stride = 1;
hidden_size = 100;
output_size = class_num;
use_num_grad = false;
network = simple_conv_net(input_dim, conv_param, hidden_size, output_size, w_decay_lambda, use_num_grad);


% set network
trainer = nn_trainer(network, x_train, t_train, x_test, t_test, opt_alg, learning_rate, ...
                 max_epoch, batch_size, verbose);


% train
tic             
[info] = trainer.train(); 
elapsedTime = toc;
fprintf('elapsed time = %5.2f [sec]\n', elapsedTime);


% plot
display_graph('epoch', 'cost', {'Tow layer net'}, {}, {info});    

train_info = info;
test_info = info;
train_info.accuracy = info.train_acc;
test_info.accuracy = info.test_acc;
display_graph('epoch', 'accuracy', {'Train', 'Test'}, {}, {train_info, test_info}); 


