clc;
clear;
close all;

rng('default')


% load dateaset
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset('mnist', './datasets/',  inf, inf, false);


% set network
network = two_layer_net(784, 50, 10, []);


% set trainer
trainer = nn_trainer(network, x_train, t_train, x_test, t_test, 'AdaGrad', 0.1, 50, 100, 1);


% train
info = trainer.train(); 


% plot
display_graph('epoch', 'cost', {'Tow layer net'}, {}, {info});    

train_info = info;
test_info = info;
train_info.accuracy = info.train_acc;
test_info.accuracy = info.test_acc;
display_graph('epoch', 'accuracy', {'Train', 'Test'}, {}, {train_info, test_info});   




