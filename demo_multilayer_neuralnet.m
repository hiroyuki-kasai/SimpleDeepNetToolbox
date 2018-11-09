clc;
clear;
close all;

rng('default')

% set parameters
max_epoch = 30;
batch_size = 100;
opt_alg = 'AdaGrad'; % 'SGD', 'AdaGrad'
learning_rate = 0.1;


% load dateaset
dataset_dir = './datasets/';
dataset_name = 'mnist';
%dataset_name = 'orl_face';
%dataset_name = 'usps';
%dataset_name = 'coil100';
%dataset_name = 'coil20';
%dataset_name = 'cifar-100';
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset(dataset_name, dataset_dir, 5000, 1000, false);


% set network
network = multilayer_neural_net(x_train, t_train, x_test, t_test, dimension, [100 80], class_num, 'relu', 'relu', 0.01, 1, 0.5, 0, 0);


% set trainer
options.opt_alg = 'AdaGrad';
options.max_epoch = 1000;
options.step_init = 0.1;
options.verbose = 1;
options.batch_size = 100;
trainer = nn_trainer(network, options);


% train
tic             
[info] = trainer.train(); 
elapsedTime = toc;
fprintf('elapsed time = %5.2f [sec]\n', elapsedTime);


% plot
fs = 20;
figure
plot(info.epoch, info.cost, '-', 'LineWidth',2,'Color', [255, 0, 0]/255);
ax1 = gca;
set(ax1,'FontSize',fs);
title('epoch vs. cost')
xlabel('epoch','FontName','Arial','FontSize',fs,'FontWeight','bold')
ylabel('cost','FontName','Arial','FontSize',fs,'FontWeight','bold')
legend('cost');

figure
plot(info.epoch, info.train_acc, '-', 'LineWidth',2,'Color', [0, 0, 255]/255); hold on 
plot(info.epoch, info.test_acc, '-', 'LineWidth',2,'Color', [0, 255, 0]/255); hold off 
ax1 = gca;
set(ax1,'FontSize',fs);
title('epoch vs. accuracy')
xlabel('epoch','FontName','Arial','FontSize',fs,'FontWeight','bold')
ylabel('accuracy','FontName','Arial','FontSize',fs,'FontWeight','bold')
legend('train', 'test');


