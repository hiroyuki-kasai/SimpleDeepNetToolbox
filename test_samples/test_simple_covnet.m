clc;
clear;
close all;

%rng('default');

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
network = simple_conv_net(input_dim, conv_param, hidden_size, output_size, w_decay_lambda, opt_alg, learning_rate);


% set network
trainer = trainer(network, x_train, t_train, x_test, t_test, ...
                 max_epoch, batch_size, use_num_grad, verbose);

% train
tic             
[info] = trainer.train(); 
elapsedTime = toc;
fprintf('elapsed time = %5.2f [sec]\n', elapsedTime);


% plot
fs = 20;
figure
plot(info.epoch_array, info.cost_array, '-', 'LineWidth',2,'Color', [255, 0, 0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
title('epoch vs. cost')
xlabel('epoch','FontName','Arial','FontSize',fs,'FontWeight','bold')
ylabel('cost','FontName','Arial','FontSize',fs,'FontWeight','bold')
legend('cost');

figure
plot(info.epoch_array, info.train_acc_array, '-', 'LineWidth',2,'Color', [0, 0, 255]/255); hold on 
plot(info.epoch_array, info.test_acc_array, '-', 'LineWidth',2,'Color', [0, 255, 0]/255); hold off 
ax1 = gca;
set(ax1,'FontSize',fs);
title('epoch vs. accuracy')
xlabel('epoch','FontName','Arial','FontSize',fs,'FontWeight','bold')
ylabel('accuracy','FontName','Arial','FontSize',fs,'FontWeight','bold')
legend('train', 'test');


