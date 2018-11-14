clc;
clear;
close all;

rng('default')


% load dateaset
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset('mnist', './datasets/',  inf, inf, false);


% set network
network = two_layer_net(x_train, t_train, x_test, t_test, 784, 50, 10, []);


% set trainer
options.opt_alg = 'SVRG';
%options.opt_alg = 'SARAH';
%options.step_init = 0.1;
options.verbose = 2;
%options.max_epoch = 20;
trainer = nn_trainer(network, options);
%trainer = nn_trainer(network);


% train
info = trainer.train(); 


% plot
display_graph('epoch', 'cost', {'Two layer net'}, {}, {info});    

train_info = info;
test_info = info;
train_info.accuracy = info.train_acc;
test_info.accuracy = info.test_acc;
display_graph('epoch', 'accuracy', {'Train', 'Test'}, {}, {train_info, test_info});   




