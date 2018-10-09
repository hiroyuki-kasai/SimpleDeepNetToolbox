clc;
clear;
close all;

%rng('default');


% set parameters
max_epoch = 30;
verbose = 1;
batchnorm_flag = 0;
%dropout_flag = 0;
%dropout_ratio = 0;
dropout_flag = 0;
dropout_ratio = 0.2;
w_decay_lambda = 0.01; % l2-norm regularization. 0.1 ? (0 ... 1)
use_num_grad = 0;
%opt_alg = 'SGD';
%opt_alg = 'Momuentum';
opt_alg = 'AdaGrad';


if 0
    total_train_size = 60000;
    total_test_size = 10000;
    batch_size = 100;
    learning_rate = 0.1;
elseif 1 % partial (overfitting situation)
    total_train_size = 3000;
    total_test_size = 3000;
    batch_size = 100;
    learning_rate = 0.1;    
else
    total_train_size = 100;
    total_test_size = 10;
    batch_size = 20; 
    learning_rate = 0.001;
end


% load dateaset
dataset_dir = '../datasets/';
%dataset_name = 'mnist';
%dataset_name = 'orl_face';
dataset_name = 'usps';
%dataset_name = 'coil100';
%dataset_name = 'coil20';
%dataset_name = 'cifar-100';
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, height, width] = ...
    load_dataset(dataset_name, dataset_dir, total_train_size, total_test_size, true);
                 



% set network
network = multilayer_neural_net(dimension, [100, 100, 100, 100, 100], class_num, 'relu', 'relu', ...
                w_decay_lambda, dropout_flag, dropout_ratio, batchnorm_flag, opt_alg, learning_rate);


% set trainer
trainer = trainer(network, x_train, t_train, x_test, t_test, ...
                 max_epoch, batch_size, use_num_grad, verbose);


%train             
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


