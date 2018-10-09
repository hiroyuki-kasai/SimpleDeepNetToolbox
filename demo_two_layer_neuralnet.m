clc;
clear;
close all;


% load dateaset
[x_train, t_train, train_num, x_test, t_test, test_num, class_num, dimension, ~, ~] = ...
    load_dataset('mnist', './datasets/',  inf, inf, false);


% set network
network = two_layer_net(784, 50, 10, 'AdaGrad', 0.1);


% set trainer
trainer = trainer(network, x_train, t_train, x_test, t_test, 30, 100, 0, 1);


% train
info = trainer.train(); 

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



